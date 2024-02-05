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
import pickle
from get_utils import get_hidden, get_force, get_hidden
from tdr import gsog
from copy import deepcopy

base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

#th._dynamo.config.cache_size_limit = 16 * 1024 ** 3  # ~ 16 GB

def train(model_num=1,ff_coefficient=0,phase='NF2',n_batch=10010,directory_name=None,loss_weight=None):
  """
  args:
  """

  interval = 100
  catch_trial_perc = 50
  all_phase = np.array(['growing_up','NF1','FF1','NF2','FF2'])
  output_folder = os.path.join(base_dir,directory_name)
  model_name = "model{:02d}".format(model_num)
  print("{}...".format(model_name))


  # find the perturbation direction
  data = get_hidden(output_folder,model_name,{'NF1':[0],'FF1':[8]},[0,8],demean=False)
  force = get_force(output_folder,model_name,{'NF1':[0],'FF1':[8]},[0,8])

  
  N_idx = 16
  F_idx = 25 # index of force
  for i in range(len(data)):
    data[i] = data[i][:,N_idx,:]
    force[i] = force[i][:,F_idx,:]
  
  # remove overall mean
  combined_N = np.vstack(data)
  mean_N = np.mean(combined_N, axis=0)

  for i in range(len(data)):
    data[i] = data[i]-mean_N
  
  # use NF1 to project onto TDR axes
  X = force[0] # NF1 force 
  N = data[0] # NF1 hidden activity

  X = np.hstack((X,np.ones((X.shape[0],1))))
  beta_b2n = np.linalg.pinv(X) @ N

  # Compute the TDR axes.
  beta_n2b = np.linalg.pinv(beta_b2n)
  beta_n2b = beta_n2b[:,:2]

  # Orthogonalize the TDR axes before projection.
  beta_n2b_orth = gsog(beta_n2b)[0]

  # uniform shift from NF1 to FF1
  us = np.mean(data[1]-data[0],axis=0)
  us = us.reshape(-1,1)
  us_orth = us - np.dot(beta_n2b_orth[:,0],us)/np.linalg.norm(beta_n2b_orth[:,0])**2 * beta_n2b_orth[:,0][:,None]
  us_orth = us_orth - np.dot(beta_n2b_orth[:,1],us_orth)/np.linalg.norm(beta_n2b_orth[:,1])**2 * beta_n2b_orth[:,1][:,None]
  us_orth_norm = us_orth/np.linalg.norm(us_orth)


  disturb_hidden=True 
  t_disturb_hidden=N_idx/100
  #d_hidden=th.from_numpy(us_orth_norm.T*4.5)
  d_hidden = th.from_numpy(us_orth_norm.T)

  weight_file = None
  cfg_file = None
  
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
  results = {}
  alphas = [-1,0,1]
  for alpha in alphas:
    results[str(alpha)] = deepcopy(losses)

  # load environment and policy
  freeze_output_layer = freeze_input_layer = (phase != 'growing_up')

  env, policy, optimizer, scheduler = load_stuff(cfg_file=cfg_file,weight_file=weight_file,phase=phase,
                                                 freeze_output_layer=freeze_output_layer,freeze_input_layer=freeze_input_layer)
  for batch in tqdm(range(n_batch), desc=f"Training {phase}", unit="batch"):

    for alpha in alphas:
      # Test the network
      pert = th.from_numpy(0.6*alpha*us_orth_norm.T)
      data, loss_test, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coefficient,loss_weight=loss_weight,
                                               disturb_hidden=True,t_disturb_hidden=t_disturb_hidden,d_hidden=pert)
      
      results[str(alpha)]['angle'].append(ang_dev.item())
      results[str(alpha)]['lateral'].append(lat_dev.item())

      for key in loss_test:
        results[str(alpha)][key].append(loss_test[key].item())

    # train the network on 8 directions only
    data = run_episode(env,policy,8,catch_trial_perc,'test',ff_coefficient=ff_coefficient,detach=False,go_cue_random=True,
                       disturb_hidden=False,t_disturb_hidden=t_disturb_hidden,d_hidden=d_hidden)
    overall_loss, _ = cal_loss(data, loss_weight=loss_weight)
    
    # update the network    
    optimizer.zero_grad()
    overall_loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    # print progress
    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, sum(losses['position'][-interval:])/interval))

      # save the result at every 1000 batches
      # save_stuff(output_folder, model_name, phase, ff_coefficient, policy, losses, env)
      #weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_batch={batch}_FFCoef={ff_coefficient}_weights")
      weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
      log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
      #log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_batch={batch}_FFCoef={ff_coefficient}_log.json")
      cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

      th.save(policy.state_dict(), weight_file)

      with open(log_file,'w') as file, open(cfg_file,'w') as cfg_file:
        json.dump(results,file)
        json.dump(env.get_save_config(),cfg_file)

  print("Done...")

def test(env,policy,ff_coefficient=0,is_channel=False,loss_weight=None,add_vis_noise=False, add_prop_noise=False, var_vis_noise=0.1, var_prop_noise=0.1,
         t_vis_noise=[0.1,0.15], t_prop_noise=[0.1,0.15],
         disturb_hidden=False,t_disturb_hidden=0.15,d_hidden=None):

  # Run episode
  data = run_episode(env, policy, batch_size=8, catch_trial_perc=0, condition='test', 
                     ff_coefficient=ff_coefficient, is_channel=is_channel, detach=True, calc_endpoint_force=True,
                     add_vis_noise=add_vis_noise, add_prop_noise=add_prop_noise, var_vis_noise=var_vis_noise, var_prop_noise=var_prop_noise,
                     t_vis_noise=t_vis_noise, t_prop_noise=t_prop_noise,
                     disturb_hidden=disturb_hidden, t_disturb_hidden=t_disturb_hidden, d_hidden=d_hidden)

  # Calculate loss
  _, loss_test = cal_loss(data,loss_weight=loss_weight)

  # anglular deviation
  ang_dev = np.mean(calculate_angles_between_vectors(data['vel'], data['tg'], data['xy']))
  lat_dev = np.mean(calculate_lateral_deviation(data['xy'], data['tg'])[0])

  return data, loss_test, ang_dev, lat_dev


def cal_loss(data, loss_weight=None,d_hidden=None):

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

  overall_loss = 0
  # projection loss:
  if d_hidden is not None:
    proj=th.mean(th.matmul(data['all_hidden'][:,15,:],d_hidden.T))
    proj_weighted = 0*proj
    overall_loss += proj_weighted
  

  if loss_weight is None:
     loss_weight = np.array([1e+3,1e+5,1e-1, 3.16227766e-04,1e-5,1e-3,0]) # 9

  loss_weighted = {
    'position': loss_weight[0]*loss['position'],
    'jerk': loss_weight[1]*loss['jerk'],
    'muscle': loss_weight[2]*loss['muscle'],
    'muscle_derivative': loss_weight[3]*loss['muscle_derivative'],
    'hidden': loss_weight[4]*loss['hidden'],
    'hidden_derivative': loss_weight[5]*loss['hidden_derivative'],
    'hidden_jerk': loss_weight[6]*loss['hidden_jerk']
  }

  
  for key in loss_weighted:
    overall_loss += loss_weighted[key]

  return overall_loss, loss_weighted


def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',
                ff_coefficient=None, is_channel=False,detach=False,calc_endpoint_force=False, go_cue_random=False,
                add_vis_noise=False, add_prop_noise=False, var_vis_noise=0.1, var_prop_noise=0.1,
                t_vis_noise=[0.1,0.15], t_prop_noise=[0.1,0.15],
                disturb_hidden=False, t_disturb_hidden=0.15, d_hidden=None):
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                        is_channel=is_channel,calc_endpoint_force=calc_endpoint_force, go_cue_random=go_cue_random)
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

    # add noise to the observation
    # vision noise: first two columns
    if add_vis_noise:
        if env.elapsed>=t_vis_noise[0] and env.elapsed<t_vis_noise[1]:
            obs[:,:2] += th.normal(0,var_vis_noise,size=(batch_size,2))
    # properioceptive noise: next 12 columns
    if add_prop_noise:
        if env.elapsed>=t_prop_noise[0] and env.elapsed<t_prop_noise[1]:
            obs[:,2:14] += th.normal(0,var_prop_noise,size=(batch_size,12))


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
    trainall = 0

    if trainall:
      directory_name = sys.argv[2]
      
      iter_list = range(1) # 20
      num_processes = len(iter_list)
      
      # with ProcessPoolExecutor(max_workers=num_processes) as executor:
      #   futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='growing_up', n_batch=20010, directory_name=directory_name, loss_weight=None): iteration for iteration in iter_list}
      #   for future in as_completed(futures):
      #     try:
      #       result = future.result()
      #     except Exception as e:
      #       print(f"Error in iteration {futures[future]}: {e}")
      
      # with ProcessPoolExecutor(max_workers=num_processes) as executor:
      #   futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF1', n_batch=2010, directory_name=directory_name, loss_weight=None): iteration for iteration in iter_list}
      #   for future in as_completed(futures):
      #     try:
      #       result = future.result()
      #     except Exception as e:
      #       print(f"Error in iteration {futures[future]}: {e}")
      
      # with ProcessPoolExecutor(max_workers=num_processes) as executor:
      #   futures = {executor.submit(train, model_num=iteration, ff_coefficient=8, phase='FF1', n_batch=10010, directory_name=directory_name, loss_weight=None): iteration for iteration in iter_list}
      #   for future in as_completed(futures):
      #     try:
      #       result = future.result()
      #     except Exception as e:
      #       print(f"Error in iteration {futures[future]}: {e}")
      

      # with ProcessPoolExecutor(max_workers=num_processes) as executor:
      #   futures = {executor.submit(train, model_num=iteration, ff_coefficient=0, phase='NF2', n_batch=7010, directory_name=directory_name, loss_weight=None): iteration for iteration in iter_list}
      #   for future in as_completed(futures):
      #     try:
      #       result = future.result()
      #     except Exception as e:
      #       print(f"Error in iteration {futures[future]}: {e}")

      with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(train, model_num=iteration, ff_coefficient=-8, phase='FF2', n_batch=10010, directory_name=directory_name, loss_weight=None): iteration for iteration in iter_list}
        for future in as_completed(futures):
          try:
            result = future.result()
          except Exception as e:
            print(f"Error in iteration {futures[future]}: {e}")
      

          
    else: ## training networks for each phase separately
      #ff_coefficient = int(sys.argv[2])
      ff_coefficient = 8

      #phase = sys.argv[3] # growing_up or anything else
      phase = 'FF2'

      #n_batch = int(sys.argv[4])
      n_batch = 2000

      #directory_name = sys.argv[5]
      directory_name = 'Sim_simple2'

      #train_single = int(sys.argv[6])
      train_single = 0

      if train_single:
        train(0,ff_coefficient,phase,n_batch=n_batch,directory_name=directory_name)
      else:
        iter_list = range(20)
        num_processes = len(iter_list)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
          futures = {executor.submit(train, iteration, ff_coefficient, phase, n_batch, directory_name): iteration for iteration in iter_list}
          for future in as_completed(futures):
            try:
              result = future.result()
            except Exception as e:
              print(f"Error in iteration {futures[future]}: {e}")