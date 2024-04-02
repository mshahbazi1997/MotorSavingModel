import os
import sys 
from utils import load_stuff
from utils import calculate_angles_between_vectors, calculate_lateral_deviation
import torch as th
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')


def train(model_num=1,ff_coefficient=0,phase='growing_up',n_batch=10010,directory_name=None,loss_weight=None,train_random=False,start_again=False,num_hidden=128):
  """
  args:
  """
  interval = 100
  catch_trial_perc = 50
  all_phase = np.array(['growing_up','NF1','FF1','NF2','FF2'])
  # create output folder if not exist
  if not os.path.exists(os.path.join(base_dir,directory_name)):
    os.makedirs(os.path.join(base_dir,directory_name))
  output_folder = os.path.join(base_dir,directory_name)
  model_name = "model{:02d}".format(model_num)
  print("{}...".format(model_name))

  if start_again:
    weight_file = None
  else:
    weight_file = next(Path(output_folder).glob(f'{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights'), None)
  #weight_file = None
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
                                                 freeze_output_layer=freeze_output_layer,freeze_input_layer=freeze_input_layer,num_hidden=num_hidden)
  batch_size = 32
  for batch in tqdm(range(n_batch), desc=f"Training {phase}", unit="batch"):

    # test the network right at the beginning
    data, loss_test, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coefficient,loss_weight=loss_weight)


    # train the network
    if train_random:
      data = run_episode(env,policy,batch_size,catch_trial_perc,'train',ff_coefficient=ff_coefficient,detach=False)
    else:
      data = run_episode(env,policy,8,catch_trial_perc,'test',ff_coefficient=ff_coefficient,detach=False,go_cue_random=True)

    overall_loss, _ = cal_loss(data, loss_weight=loss_weight)
    
    # update the network    
    optimizer.zero_grad()
    overall_loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    # Save losses
    losses['overall'].append(overall_loss.item())
    for key in loss_test:
      if key == 'position':
        temp=[]
        for jj in range(8):
          temp.append(loss_test[key][jj].item())
        losses[key].append(temp)
      else:
        losses[key].append(loss_test[key].item())

    temp=[]
    for jj in range(8):
      temp.append(ang_dev[jj].item())
    losses['angle'].append(temp)

    temp=[]
    for jj in range(8):
      temp.append(lat_dev[jj].item())
    losses['lateral'].append(temp)

    #losses['angle'].append(ang_dev.item())
    #losses['lateral'].append(lat_dev.item())

    # print progress
    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, np.mean(sum(np.array(losses['position'])[-interval:,:])/interval)))

      weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
      log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
      cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

      th.save(policy.state_dict(), weight_file)

      with open(log_file,'w') as file, open(cfg_file,'w') as cfg_file:
        json.dump(losses,file)
        json.dump(env.get_save_config(),cfg_file)

  print("Done...")


def test(env,policy,ff_coefficient=0,is_channel=False,
         batch_size=8, catch_trial_perc=0, condition='test', go_cue_random=None,
         loss_weight=None,disturb_hidden=False,t_disturb_hidden=0.15,d_hidden=None, seed=None):

  # Run episode
  data = run_episode(env, policy, batch_size=batch_size, catch_trial_perc=catch_trial_perc, condition=condition,
                     go_cue_random=go_cue_random,
                     ff_coefficient=ff_coefficient, is_channel=is_channel, detach=True, calc_endpoint_force=True,
                     disturb_hidden=disturb_hidden, t_disturb_hidden=t_disturb_hidden, d_hidden=d_hidden, seed=seed)

  # Calculate loss
  _, loss_test = cal_loss(data,loss_weight=loss_weight)

  # anglular deviation
  #ang_dev = np.mean(calculate_angles_between_vectors(data['vel'], data['tg'], data['xy']))
  #lat_dev = np.mean(calculate_lateral_deviation(data['xy'], data['tg'])[0])
  ang_dev = calculate_angles_between_vectors(data['vel'], data['tg'], data['xy'])
  lat_dev = calculate_lateral_deviation(data['xy'], data['tg'])[0]

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

  
  loss['position'] = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1),axis=1) # average over time not targets
  loss['jerk'] = th.mean(th.sum(th.square(th.diff(data['vel'], n=2, dim=1)), dim=-1))
  loss['muscle'] = th.mean(th.sum(data['all_force'], dim=-1))
  loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], n=1, dim=1)), dim=-1))
  loss['hidden'] = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  loss['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=1, dim=1)), dim=-1))
  loss['hidden_jerk'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=3, dim=1)), dim=-1))
  

  if loss_weight is None:
     # currently in use
     loss_weight = np.array([1e+3,1e+5,1e-1, 3.16227766e-04,1e-5,1e-3,0])
     
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
    if key=='position':
      overall_loss += th.mean(loss_weighted[key])
    else:
      overall_loss += loss_weighted[key]

  return overall_loss, loss_weighted


def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',
                ff_coefficient=None, is_channel=False,detach=False,calc_endpoint_force=False, go_cue_random=None,
                disturb_hidden=False, t_disturb_hidden=0.15, d_hidden=None, seed=None):
  
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                        is_channel=is_channel,calc_endpoint_force=calc_endpoint_force, go_cue_random=go_cue_random, seed = seed)
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
    action, h = policy(obs,h)
    obs, terminated, info = env.step(action=action)

    # add disturn hidden activity
    if disturb_hidden:
      if env.elapsed==t_disturb_hidden:
        dh = d_hidden.repeat(1,batch_size,1)
        h += dh
    
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
    
    #trainall = int(sys.argv[1])
    trainall = 1

    if trainall:


      for network_siz in [128,256]:
        #directory_name = sys.argv[2]
        directory_name = f'Sim_simp_{network_siz}'
        num_hidden = network_siz
        
        iter_list = range(40) # 20
        num_processes = len(iter_list)
        #n_batches = [20010,401,3201,1301,3201] # for Sim_simple_XX - old
        n_batches = [20010,10001,3201,10001,3201] # for Sim_simple_XX
        #n_batches = [20010,2001,10001,7001,10001] # for Sim_all_XX
        train_random = False
        
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='growing_up', n_batch=n_batches[0], directory_name=directory_name, loss_weight=None, train_random=True,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF1', n_batch=n_batches[1], directory_name=directory_name, loss_weight=None, train_random=train_random,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=8, phase='FF1', n_batch=n_batches[2], directory_name=directory_name, loss_weight=None, train_random=train_random,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures): 
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
        

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF2', n_batch=n_batches[3], directory_name=directory_name, loss_weight=None, train_random=train_random,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures): 
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, model_num=iteration, ff_coefficient=8, phase='FF2', n_batch=n_batches[4], directory_name=directory_name, loss_weight=None, train_random=train_random,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures): 
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")
      

          
    else: ## training networks for each phase separately
      #ff_coefficient = int(sys.argv[2])
      #phase = sys.argv[3] # growing_up or anything else
      #n_batch = int(sys.argv[4])
      #directory_name = sys.argv[5]
      #train_single = int(sys.argv[6])


      phase = 'FF2'
      ff_coefficient = 8
      n_batch = 101
      start_again = True

      directory_name = 'Sim_simple_32'
      num_hidden = 32

      train_single = 1
      train_random = False
      
      

      if train_single:
        train(0,ff_coefficient,phase,n_batch=n_batch,directory_name=directory_name,train_random=train_random,start_again=start_again,num_hidden=num_hidden)
      else:
        iter_list = range(20)
        num_processes = len(iter_list)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, iteration, ff_coefficient, phase, n_batch, directory_name, train_random=train_random,start_again=start_again,num_hidden=num_hidden): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")