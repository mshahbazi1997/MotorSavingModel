import os
import sys 
from utils import create_directory, load_env
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



def train(model_num,ff_coefficient,phase,n_batch=10000,directory_name=None):

  device = th.device("cpu")
  interval = 1000
  num_hidden = 128
  catch_trial_perc = 50
  all_phase = np.array(['growing_up','NF1','FF1','NF2','FF2'])
  

  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  print("{}...".format(model_name))

  
  if phase=='growing_up':
    env = load_env(CentreOutFF)
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device)

    optimizer = th.optim.Adam(policy.parameters(), lr=0.001,eps=1e-7)
    batch_size = 128

  else:
    phase_prev = 'growing_up' if phase not in all_phase else all_phase[all_phase.tolist().index(phase) - 1]
    weight_file, cfg_file = (next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_weights')),
                             next(Path(output_folder).glob(f'{model_name}_phase={phase_prev}_*_cfg.json')))
    cfg = json.load(open(cfg_file, 'r'))
    env = load_env(CentreOutFF, cfg)
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, freeze_output_layer=True)
    policy.load_state_dict(th.load(weight_file))

    optimizer = th.optim.SGD(policy.parameters(), lr=0.001)
    batch_size = 1024


  losses = {
    'overall': [],
    'position': [],
    'angle': [],
    'lateral': [],
    'muscle': [],
    'hidden': []}
  
  
  for batch in tqdm(range(n_batch), desc=f"Training {phase}", unit="batch"):

    # Run episode
    data = run_episode(env,policy,batch_size,catch_trial_perc,'train',ff_coefficient=ff_coefficient,detach=False)

    # calculate losses
    loss, _, muscle_loss, hidden_loss, _, _ = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=False)
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()

    # TEST
    # Run episode
    data = run_episode(env,policy,8,0,'test',ff_coefficient=ff_coefficient,detach=True)

    # calculate losses
    _, position_loss, _, _, angle_loss, lateral_loss = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=True)


    # Update loss values in the dictionary
    losses['overall'].append(loss.item())
    losses['position'].append(position_loss.item())
    losses['angle'].append(angle_loss.item())
    losses['lateral'].append(lateral_loss.item())
    losses['muscle'].append(muscle_loss.item())
    losses['hidden'].append(hidden_loss.item())

    # print progress
    #if (batch % interval == 0) and (batch != 0):
    #  print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, sum(losses['position'][-interval:])/interval))

  # save weights and losses
  weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
  log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
  cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

  th.save(policy.state_dict(), weight_file)

  with open(log_file,'w') as file, open(cfg_file,'w') as cfg_file:
    json.dump(losses,file)
    json.dump(env.get_save_config(),cfg_file)

  print("Done...")


def test(cfg_file,weight_file,ff_coefficient=None,is_channel=False,K=1,B=-1):
  device = th.device("cpu")

  # load configuration
  cfg = json.load(open(cfg_file, 'r'))

  if ff_coefficient is None:
    ff_coefficient=cfg['ff_coefficient']
    
  # environment and network
  env = load_env(CentreOutFF, cfg)
  w = th.load(weight_file)
  num_hidden = int(w['gru.weight_ih_l0'].shape[0]/3)
  if 'h0' in w.keys():
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=True)
  else:
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=False)
  policy.load_state_dict(w)
  
  
  # Run episode
  data = run_episode(env,policy,8,0,'test',ff_coefficient=ff_coefficient,is_channel=is_channel,K=K,B=B,detach=True)
  

  return data


def cal_loss(data, max_iso_force, dt, policy, test=False):

  # calculate losses
  # # input_loss
  # input_loss = th.sqrt(th.sum(th.square(policy.gru.weight_ih_l0)))
  # # muscle_loss
  # max_iso_force_n = max_iso_force / th.mean(max_iso_force) 
  # y = data['all_muscle'] * max_iso_force_n
  # muscle_loss = th.mean(th.square(y))
  # # hidden_loss
  # y = data['all_hidden']
  # dy = th.diff(y,axis=1)/dt
  # hidden_loss = th.mean(th.square(y))+0.05*th.mean(th.square(dy))
  # # position_loss
  # position_loss = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
  # # recurrent_loss
  # #recurrent_loss = th.sqrt(th.sum(th.square(policy.gru.weight_hh_l0)))

  # loss = 1e-6*input_loss + 20*muscle_loss + 0.1*hidden_loss + 2*position_loss #+ 1e-5*recurrent_loss

  # Jon's proposed loss function
  position_loss = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
  muscle_loss = th.mean(th.sum(data['all_force'], dim=-1))
  hidden_loss = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  diff_loss =  th.mean(th.sum(th.square(th.diff(data['all_hidden'], 1, dim=1)), dim=-1))

  loss = position_loss + 1e-3*muscle_loss + 5e-5*hidden_loss + 3e-2*diff_loss
  #loss = position_loss + 1e-4*muscle_loss + 5e-5*hidden_loss + 1e-1*diff_loss

  angle_loss = None
  lateral_loss = None
  if test:
    angle_loss = np.mean(calculate_angles_between_vectors(data['vel'], data['tg'], data['xy']))
    lateral_loss, _, _, _ = calculate_lateral_deviation(data['xy'], data['tg'])
    lateral_loss = np.mean(lateral_loss)

  return loss, position_loss, muscle_loss, hidden_loss, angle_loss, lateral_loss



def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',ff_coefficient=None, is_channel=False,K=1,B=-1,detach=False):
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, is_channel=is_channel,K=K,B=B)
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
  }

  while not terminated:
      # Append data to respective lists
      data['all_hidden'].append(h[0, :, None, :])
      data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])
      data['all_endpoint'].append(info['endpoint_load'][:, None, :])

      action, h = policy(obs, h)
      obs, _, terminated, _, info = env.step(action=action)

      data['xy'].append(info["states"]["fingertip"][:, None, :])
      data['tg'].append(info["goal"][:, None, :])
      data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
      data['all_actions'].append(action[:, None, :])
      data['all_force'].append(info['states']['muscle'][:, 6, None, :])

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
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'growing_up',n_batch=10000,directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'NF1',n_batch=500,directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,'FF1',n_batch=3000,directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,'NF2',n_batch=500,directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,'FF2',n_batch=3000,directory_name=directory_name) 
                                                     for iteration in these_iters)
          
    else: ## training networks for each phase separately
      ff_coefficient = int(sys.argv[2])
      phase = sys.argv[3] # growing_up or anything else
      n_batch = int(sys.argv[4])
      directory_name = sys.argv[5]

      iter_list = range(16)
      n_jobs = 16

      #train(1,ff_coefficient,phase,n_batch=n_batch,directory_name=directory_name)
      while len(iter_list) > 0:
         these_iters = iter_list[0:n_jobs]
         iter_list = iter_list[n_jobs:]
         result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,ff_coefficient,phase,n_batch=n_batch,directory_name=directory_name) 
                                                     for iteration in these_iters)

