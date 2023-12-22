import os
import sys 
from utils import create_directory, load_stuff
from utils import calculate_angles_between_vectors, calculate_lateral_deviation
import torch as th
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product




#th._dynamo.config.cache_size_limit = 16 * 1024 ** 3  # ~ 16 GB

def train(model_num=1,ff_coefficient=0,phase='growing_up',n_batch=10010,directory_name=None,loss_weight=None):
  """
  args:
  """

  interval = 200
  catch_trial_perc = 50
  all_phase = np.array(['growing_up','NF1','FF1','NF2','FF2'])
  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  print("{}...".format(model_name))


  # check if we have the file already related to this phase
  # if so, the continue training from there
  weight_file = next(Path(output_folder).glob(f'{model_name}_phase={phase}_*_weights'), None)
  weight_file = None # Delete this later
  if weight_file is not None:
    cfg_file = next(Path(output_folder).glob(f'{model_name}_phase={phase}_*_cfg.json'))

    # open loss file
    loss_file = next(Path(output_folder).glob(f'{model_name}_phase={phase}_*_log.json'))
    losses = json.load(open(loss_file,'r'))
  else:
    weight_file = None
    cfg_file = None
    if phase != 'growing_up':
      phase_prev = 'growing_up' if phase not in all_phase else all_phase[all_phase.tolist().index(phase) - 1]
      weight_file, cfg_file = (next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_weights')),
                               next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_cfg.json')))

    losses = {
      'overall': [],
      'position': [],
      'jerk': [],
      'muscle': [],
      'muscle_derivative': [],
      'hidden': [],
      'hidden_derivative': [],
      'hidden_jerk': [],
      'angle': [],
      'lateral': []
      }

  # load environment and policy
  freeze_output_layer = freeze_input_layer = (phase != 'growing_up')

  env, policy, optimizer, scheduler = load_stuff(cfg_file=cfg_file,weight_file=weight_file,phase=phase,
                                                 freeze_output_layer=freeze_output_layer,freeze_input_layer=freeze_input_layer)
  batch_size = 32

  for batch in tqdm(range(n_batch), desc=f"Training {phase}", unit="batch"):

    # train the network
    data = run_episode(env,policy,batch_size,catch_trial_perc,'train',ff_coefficient=ff_coefficient,detach=False)
    overall_loss, _ = cal_loss(data, loss_weight=loss_weight)
    
    
    optimizer.zero_grad() 
    overall_loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    # test the network
    data, loss_test, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coefficient,loss_weight=loss_weight)
    
    # Save losses
    losses['overall'].append(overall_loss.item())
    for key in loss_test:
      losses[key].append(loss_test[key].item())
    
    losses['angle'].append(ang_dev.item())
    losses['lateral'].append(lat_dev.item())

    # print progress
    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, sum(losses['position'][-interval:])/interval))

      # save the result at every 1000 batches
      # save_stuff(output_folder, model_name, phase, ff_coefficient, policy, losses, env)
      weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
      log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
      cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

      th.save(policy.state_dict(), weight_file)

      with open(log_file,'w') as file, open(cfg_file,'w') as cfg_file:
        json.dump(losses,file)
        json.dump(env.get_save_config(),cfg_file)
    

  print("Done...")


def test(env,policy,ff_coefficient=0,is_channel=False,loss_weight=None):

  # Run episode
  data = run_episode(env, policy, batch_size=8, catch_trial_perc=0, condition='test', ff_coefficient=ff_coefficient, is_channel=is_channel, detach=True, calc_endpoint_force=True)

  # Calculate loss
  _, loss_test = cal_loss(data,loss_weight=loss_weight)

  # anglular deviation
  ang_dev = np.mean(calculate_angles_between_vectors(data['vel'], data['tg'], data['xy']))
  lat_dev = np.mean(calculate_lateral_deviation(data['xy'], data['tg'])[0])

  return data, loss_test, ang_dev, lat_dev


def cal_loss(data, loss_weight=None):

  loss = {
    'position': None,
    'jerk': None,
    'muscle': None,
    'muscle_derivative': None,
    'hidden': None,
    'hidden_derivative': None,
    'hidden_jerk': None,}

  
  loss['position'] = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
  loss['jerk'] = th.mean(th.sum(th.square(th.diff(data['vel'], n=2, dim=1)), dim=-1))
  loss['muscle'] = th.mean(th.sum(data['all_force'], dim=-1))
  loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], n=1, dim=1)), dim=-1))
  loss['hidden'] = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  loss['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=1, dim=1)), dim=-1))
  loss['hidden_jerk'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=3, dim=1)), dim=-1))
  

  if loss_weight is None:
     # position, jerk, muscle, muscle_derivative, hidden, hidden_derivative, hidden_jerk
     #loss_weight = [1, 2e2, 1e-4, 1e-5, 3e-5, 2e-2, 0] # Mahdiyar's OLD
     #loss_weight = [1, 1e1, 1e-4, 1e-5, 3e-5, 2e-2, 0] # Mahdiyar's version growing_up
     loss_weight = [1e1, 2e3, 1e-4, 1e-5, 3e-5, 2e-2, 0] # FF1 GOOD not straight trajectory
     loss_weight = [1e2, 3e3, 1e-4, 1e-5, 3e-5, 2e-2, 0] # FF1 GOOD but large muscle activity
     loss_weight = [1e1, 2e3, 1e-4, 1e-5, 3e-5, 2e-2, 0] # 


     #loss_weight = [1e2, 1e-6*1e8, 1e-1, 0, 1e-2, 0, 2e-10*1e12] # Jon's version
     #loss_weight = [1e3, 1e-7**1e12, 5e-2, 1e-8*1e4, 1e-4, 0, 1e-8*1e12] # Pauls's version (very large muscle activity)

     
  loss_weighted = {
    'position': loss_weight[0]*loss['position'],
    'jerk': loss_weight[1]*loss['jerk'],
    'muscle': loss_weight[2]*loss['muscle'],
    'muscle_derivative': loss_weight[3]*loss['muscle_derivative'],
    'hidden': loss_weight[4]*loss['hidden'],
    'hidden_derivative': loss_weight[5]*loss['hidden_derivative'],
    'hidden_jerk': loss_weight[6]*loss['hidden_jerk']
  }

  overall_loss = 0
  for key in loss_weighted:
    overall_loss += loss_weighted[key]

  return overall_loss, loss_weighted


def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',ff_coefficient=None, is_channel=False,detach=False,calc_endpoint_force=False):
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                        is_channel=is_channel,calc_endpoint_force=calc_endpoint_force)
  terminated = False

  # Initialize a dictionary to store lists
  data = {
      'xy': [],
      'tg': [],
      'vel': [],
      'all_action': [],
      'all_hidden': [],
      'all_muscle': [],
      'all_force': [],
      'endpoint_load': [],
      'endpoint_force': []
  }

  while not terminated:

      action, h = policy(obs, h)
      obs, terminated, info = env.step(action=action)


      data['all_hidden'].append(h[0, :, None, :])
      data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])
      data['all_force'].append(info['states']['muscle'][:, -1, None, :])
      data['xy'].append(info["states"]["fingertip"][:, None, :])
      data['tg'].append(info["goal"][:, None, :])
      data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
      data['all_action'].append(action[:, None, :])
      data['endpoint_load'].append(info['endpoint_load'][:, None, :])
      data['endpoint_force'].append(info['endpoint_force'][:, None, :])
      

  # Concatenate the lists
  for key in data:
      data[key] = th.cat(data[key], axis=1)

  if detach:
      # Detach tensors if needed
      for key in data:
          data[key] = th.detach(data[key])

  return data


if __name__ == "__main__":
    
    trainall = int(sys.argv[1])

    if trainall:
      #directory_name = sys.argv[2]


      loss_weights = np.array([1e+3,   # position
                                 1e+5,   # jerk
                                 1e-1,   # muscle
                                 1e-5,   # muscle_derivative
                                 3e-5,   # hidden 
                                 2e-2,   # hidden_derivative
                                 0])     # hidden_jerk
      md_w = np.logspace(start=-6,stop=-1,num=3)
      h_w = np.logspace(start=-5,stop=-2,num=3)
      hd_w = np.logspace(start=-3,stop=-1,num=3)

      mhh_w = np.array(list(product(md_w,h_w,hd_w)))

      iter_list = range(20)
      num_processes = len(iter_list)

      lw = [loss_weights.copy() for _ in range(len(mhh_w))]

      for idx,mhhw in enumerate(mhh_w):
        lw[idx][3] = mhhw[0]
        lw[idx][4] = mhhw[1]
        lw[idx][5] = mhhw[2]

      # these loss i want to try [9,10,11,16,17,18,19,20,21,23,26]
        
      for j in [9,16,10,11,17,19,20,21,23,26]:
        loss_weight = lw[j]
        directory_name = f"loss{str(j)}"
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='growing_up', n_batch=20010, directory_name=directory_name, loss_weight=loss_weight): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF1', n_batch=2010, directory_name=directory_name, loss_weight=loss_weight): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=8, phase='FF1', n_batch=10010, directory_name=directory_name, loss_weight=loss_weight): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF2', n_batch=7010, directory_name=directory_name, loss_weight=loss_weight): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=8, phase='FF2', n_batch=10010, directory_name=directory_name, loss_weight=loss_weight): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        

          
    else: ## training networks for each phase separately
      ff_coefficient = int(sys.argv[2])
      phase = sys.argv[3] # growing_up or anything else
      n_batch = int(sys.argv[4])
      directory_name = sys.argv[5]
      train_single = int(sys.argv[6])

      if train_single:
        train(0,ff_coefficient,phase,n_batch=n_batch,directory_name=directory_name)
      else:
        loss_weights = np.array([1e+3,   # position
                                 1e+5,   # jerk
                                 1e-1,   # muscle
                                 1e-5,   # muscle_derivative
                                 3e-5,   # hidden 
                                 2e-2,   # hidden_derivative
                                 0])     # hidden_jerk
        md_w = np.logspace(start=-6,stop=-1,num=3)
        h_w = np.logspace(start=-5,stop=-2,num=3)
        hd_w = np.logspace(start=-3,stop=-1,num=3)

        mhh_w = np.array(list(product(md_w,h_w,hd_w)))

        iter_list = range(len(mhh_w))
        num_processes = len(iter_list)

        lw = [loss_weights.copy() for _ in range(len(mhh_w))]

        for idx,mhhw in enumerate(mhh_w):
          lw[idx][3] = mhhw[0]
          lw[idx][4] = mhhw[1]
          lw[idx][5] = mhhw[2]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, iteration, ff_coefficient, phase, n_batch, directory_name, lw[iteration]): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")