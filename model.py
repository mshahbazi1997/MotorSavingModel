import os
import sys 
from utils import create_directory, load_env, load_policy
from utils import calculate_angles_between_vectors, calculate_lateral_deviation
import motornet as mn
from task import CentreOutFF
from policy import Policy
import torch as th
import numpy as np
import json
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from motornet.policy import ModularPolicyGRU




def train(model_num=1,ff_coefficient=0,phase='growing_up',continue_train=0,n_batch=50000,directory_name=None,loss_weight=None,modular=0):
  """
  args:
    continue_train (int 0/1):
      specifies whether go to the next phase or continue training the network
  """

  device = th.device("cpu")
  interval = 1000
  catch_trial_perc = 50
  all_phase = np.array(['growing_up','NF1','FF1','NF2','FF2','NF3','FF3'])
  

  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  print("{}...".format(model_name))

  if continue_train==0:
    if phase=='growing_up':
      env = load_env(CentreOutFF)
      policy = load_policy(env,modular=modular)

      optimizer = th.optim.Adam(policy.parameters(), lr=0.001,eps=1e-7)
      batch_size = 128
      pert_prob = 0 # 50
      losses = {
         'overall': [],
         'position': [],
         'angle': [],
         'lateral': [],
         'muscle': [],
         'hidden': [],
         'hidden_derivative': [],
         'muscle_derivative': [],
         'jerk': []}
    else:
      pert_prob = 0
      phase_prev = 'growing_up' if phase not in all_phase else all_phase[all_phase.tolist().index(phase) - 1]
      weight_file, cfg_file = (next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_weights')),
                              next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_cfg.json')))
      cfg = json.load(open(cfg_file,'r'))
      env = load_env(CentreOutFF,cfg)


      policy = load_policy(env,modular=modular,freeze_output_layer=False, freeze_input_layer=False,freeze_bias_hidden=False, freeze_h0=False)

      policy.load_state_dict(th.load(weight_file))

      #optimizer = th.optim.SGD(policy.parameters(), lr=0.005)
      optimizer = th.optim.Adam(policy.parameters(), lr=0.001,eps=1e-7)
      batch_size = 128 # 200
      losses = {
         'overall': [],
         'position': [],
         'angle': [],
         'lateral': [],
         'muscle': [],
         'hidden': [],
         'hidden_derivative': [],
         'muscle_derivative': [],
         'jerk': []}
  else:
    pert_prob = 0
    # currently only for continue training in NF1/FF1/NF2/FF2
    weight_file, cfg_file = (next(Path(output_folder).glob(f'{model_name}_phase={phase}_*_weights')),
                            next(Path(output_folder).glob(f'{model_name}_phase={phase}_*_cfg.json')))
    cfg = json.load(open(cfg_file,'r'))
    env = load_env(CentreOutFF,cfg)

    policy = load_policy(env,modular=modular,freeze_output_layer=True, freeze_input_layer=True,freeze_bias_hidden=False, freeze_h0=False)

    policy.load_state_dict(th.load(weight_file))

    optimizer = th.optim.SGD(policy.parameters(), lr=0.005)
    batch_size = 200

    # load losses to attach
    log_file = list(Path(output_folder).glob(f'{model_name}_phase={phase}_*_log.json'))[0]
    losses = json.load(open(log_file,'r'))


  #for batch in range(n_batch):
  for batch in tqdm(range(n_batch), desc=f"Training {phase}", unit="batch"):

    # Run episode
    data = run_episode(env,policy,batch_size,catch_trial_perc,'train',ff_coefficient=ff_coefficient,detach=False,pert_prob=pert_prob)

    # calculate losses
    #loss_train = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=False,loss_weight=loss_weight)
    loss_train = cal_loss(data, test=False, loss_weight=loss_weight)
    # backward pass & update weights
    optimizer.zero_grad() 
    loss_train['overall'].backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()

    # TEST
    # Run episode
    data = run_episode(env,policy,8,0,'test',ff_coefficient=ff_coefficient,detach=True)

    # calculate losses
    #loss_test = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=True,loss_weight=loss_weight)
    loss_test = cal_loss(data,test=True,loss_weight=loss_weight)


    # Update loss values in the dictionary
    losses['overall'].append(loss_train['overall'].item())
    losses['position'].append(loss_test['position'].item())
    losses['angle'].append(loss_test['angle'].item())
    losses['lateral'].append(loss_test['lateral'].item())
    losses['muscle'].append(loss_train['muscle'].item())
    losses['hidden'].append(loss_train['hidden'].item())

    losses['hidden_derivative'].append(loss_test['hidden_derivative'].item())
    losses['muscle_derivative'].append(loss_test['muscle_derivative'].item())
    losses['jerk'].append(loss_test['jerk'].item())

    # print progress
    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, sum(losses['position'][-interval:])/interval))

      # save the result at every 1000 batches
      weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
      log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
      cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

      th.save(policy.state_dict(), weight_file)

      with open(log_file,'w') as file, open(cfg_file,'w') as cfg_file:
        json.dump(losses,file)
        json.dump(env.get_save_config(),cfg_file)

  print("Done...")


def test(cfg_file,weight_file,ff_coefficient=None,is_channel=False,K=1,B=-1,dT=None,calc_endpoint_force=False):
  modular = 1
  device = th.device("cpu")

  # load configuration
  cfg = json.load(open(cfg_file, 'r'))

  if ff_coefficient is None:
    ff_coefficient=cfg['ff_coefficient']
    
  # environment and network
  env = load_env(CentreOutFF, cfg, dT=dT)
  w = th.load(weight_file)

  # load policy
  policy = load_policy(env,modular=modular)
  policy.load_state_dict(w)
  
  
  # Run episode
  data = run_episode(env,policy,8,0,'test',ff_coefficient=ff_coefficient,is_channel=is_channel,K=K,B=B,detach=True,pert_prob=0,calc_endpoint_force=calc_endpoint_force)
  

  return data


def cal_loss(data, loss_weight=None, test=False):
  # data, max_iso_force, dt, policy, loss_weight=None, test=False

  loss = {
    'overall': None,
    'position': None,
    'angle': None,
    'lateral': None,
    'muscle': None,
    'hidden': None,
    'jerk': None,
    'input': None,
    'recurrent': None,
    'muscle_derivative': None,
    'hidden_derivative': None}

  # Another loss version - this one is good enough (tend to produce slower movements)
  
  loss['position'] = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
  loss['muscle'] = th.mean(th.sum(data['all_force'], dim=-1))
  loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], 1, dim=1)), dim=-1))
  loss['hidden'] = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  loss['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], 1, dim=1)), dim=-1))
  loss['jerk'] = th.mean(th.sum(th.square(th.diff(data['vel'],n=2,dim=1)), dim=-1))

  if loss_weight is None:
     #loss_weight = [1, 1e-4, 1e-4, 5e-5, 3e-2, 0] # this one works very nicely
     # position, muscle, muscle_derivative, hidden, hidden_derivative, jerk
     loss_weight = [1, 1e-4, 1e-5, 3e-5, 2e-2, 2e2] 


  loss['overall'] = \
   loss_weight[0]*loss['position'] + \
   loss_weight[1]*loss['muscle'] + \
   loss_weight[2]*loss['muscle_derivative'] + \
   loss_weight[3]*loss['hidden'] + \
   loss_weight[4]*loss['hidden_derivative'] + \
   loss_weight[5]*loss['jerk']

  if test:
    # angle_loss
    loss['angle'] = np.mean(calculate_angles_between_vectors(data['vel'], data['tg'], data['xy']))

    # lateral_loss
    out = calculate_lateral_deviation(data['xy'], data['tg'])
    loss['lateral'] = np.mean(out[0])

  return loss


def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',ff_coefficient=None, is_channel=False,K=1,B=-1,detach=False,pert_prob=0.0,calc_endpoint_force=False):
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, is_channel=is_channel,K=K,B=B,pert_prob=pert_prob,calc_endpoint_force=calc_endpoint_force)
  terminated = False

  # Initialize a dictionary to store lists
  data = {
      'xy': [],
      'tg': [],
      'vel': [],
      'all_actions': [],
      'all_hidden': [],
      'all_muscle': [],
      'all_force': [],
      'all_endpoint': [],
      'endpoint_force': []
  }

  while not terminated:
      # Append data to respective lists
      if len(h.shape)==2:
        data['all_hidden'].append(h[:, None, :])
      else:
        data['all_hidden'].append(h[0, :, None, :])      
      data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])
      data['all_force'].append(info['states']['muscle'][:, 6, None, :])
      data['all_endpoint'].append(info['endpoint_load'][:, None, :])
      data['endpoint_force'].append(info['endpoint_force'][:, None, :])

      action, h = policy(obs, h)
      obs, _, terminated, _, info = env.step(action=action)

      data['xy'].append(info["states"]["fingertip"][:, None, :])
      data['tg'].append(info["goal"][:, None, :])
      data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
      data['all_actions'].append(action[:, None, :])
      

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
      directory_name = sys.argv[2]

      iter_list = range(16)
      n_jobs = 16
      while len(iter_list) > 0:
          these_iters = iter_list[0:n_jobs]
          iter_list = iter_list[n_jobs:]
          # pretraining the network using ADAM
          #result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'growing_up',n_batch=50000,directory_name=directory_name) 
          #                                           for iteration in these_iters)
          # NF1
          #result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'NF1',n_batch=30000,directory_name=directory_name) 
          #                                           for iteration in these_iters)
          # FF1
          #result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,'FF1',n_batch=60000,directory_name=directory_name) 
          #                                           for iteration in these_iters)
          # NF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'NF2',n_batch=60000,directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,'FF2',n_batch=60000,directory_name=directory_name) 
                                                     for iteration in these_iters)
          
    else: ## training networks for each phase separately
      ff_coefficient = int(sys.argv[2])
      phase = sys.argv[3] # growing_up or anything else
      n_batch = int(sys.argv[4])
      directory_name = sys.argv[5]
      continue_train = int(sys.argv[6]) if len(sys.argv) > 6 else 0


      iter_list = range(16)
      n_jobs = 16


      #loss_weight = np.array([[1, 1e-4, 1e-5, 3e-5, 2e-2, 2e2],
      #                        [1, 1e-4, 4e-6, 3e-5, 2e-2, 2e2],
      #                        [1, 1e-4, 5e-5, 3e-5, 2e-2, 2e2]])

      #train(0,ff_coefficient,phase,continue_train=continue_train,n_batch=n_batch,directory_name=directory_name)
      while len(iter_list) > 0:
         these_iters = iter_list[0:n_jobs]
         iter_list = iter_list[n_jobs:]
         result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,ff_coefficient,phase,continue_train=continue_train,n_batch=n_batch,directory_name=directory_name)  # ,loss_weight=loss_weight[1]
                                                     for iteration in these_iters)

