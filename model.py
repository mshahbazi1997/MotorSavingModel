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

def train(model_num,ff_coefficient,phase,condition='train',directory_name=None):
  num_hidden=50
  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  device = th.device("cpu")

  # Set configuaration and network
  if phase>=1:
    print("Training phase {}...".format(phase))
    # load config and weights from the previous phase
    weight_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_weights'))[0]
    cfg_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_cfg.json'))[0]

    # load configuration
    with open(cfg_file,'r') as file:
      cfg = json.load(file)

    # environment and network
    env = load_env(CentreOutFF,cfg)
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, freeze_output_layer=True)
    policy.load_state_dict(th.load(weight_file))

  else:
    # environment and network
    env = load_env(CentreOutFF)
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device)
  
  
  if condition=='growing_up': 
    optimizer = th.optim.Adam(policy.parameters(), lr=0.001,eps=1e-7)
    batch_size = 128
    catch_trial_perc = 50
    n_batch = 10000

  else: # for training use biologily plausible optimizer
    optimizer = th.optim.SGD(policy.parameters(), lr=0.001)
    batch_size = 1024
    catch_trial_perc = 50
    n_batch = 500 # don't need such long training for phase 1 and 3
    if phase == 2 or phase == 4:
      n_batch = 3000
    
  # Define Loss function
  def l1(x, y, target_size=0.01):
    """L1 loss"""
    #mask = th.norm(x-y,dim=-1,p='fro')<target_size
    #loss_ = th.sum(th.abs(x-y), dim=-1)
    #loss_[mask] = 0
    #return th.mean(loss_)
    return th.mean(th.sum(th.abs(x - y), dim=-1))

  # Train network
  overall_losses = []
  position_losses = []
  angle_losses = []
  lat_losses = []
  muscle_losses = []
  hidden_losses = []
  interval = 1000

  for batch in range(n_batch):

    # Run episode
    h = policy.init_hidden(batch_size=batch_size)

    obs, info = env.reset(condition='train',catch_trial_perc=catch_trial_perc,ff_coefficient=ff_coefficient, options={'batch_size':batch_size})
    terminated = False

    # initial positions and targets
    xy = []
    tg = []
    all_actions = []
    all_hidden = []
    all_muscle = []
    all_force = []

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
      
      action, h = policy(obs,h)
      all_hidden.append(h[0,:,None,:])
      obs, _, terminated, _, info = env.step(action=action)

      xy.append(info['states']['cartesian'][:, None, :])  # trajectories
      tg.append(info["goal"][:, None, :])  # targets
      all_actions.append(action[:, None, :])
      all_muscle.append(info['states']['muscle'][:,0,None,:])
      all_force.append(info['states']['muscle'][:,6,None,:])

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=1)
    tg = th.cat(tg, axis=1)
    
    all_actions = th.cat(all_actions, axis=1)
    all_muscle = th.cat(all_muscle, axis=1)
    all_hidden = th.cat(all_hidden, axis=1)
    all_force = th.cat(all_force, axis=1)

    # calculate losses
    # input_loss
    input_loss = th.sqrt(th.sum(th.square(policy.gru.weight_ih_l0)))
    # muscle_loss
    max_iso_force_n = env.muscle.max_iso_force / th.mean(env.muscle.max_iso_force) 
    y = all_muscle * max_iso_force_n
    muscle_loss = th.mean(th.square(y))
    # hidden_loss
    y = all_hidden
    dy = th.diff(y,axis=1)/env.dt
    hidden_loss = th.mean(th.square(y))+0.05*th.mean(th.square(dy))
    # position_loss
    position_loss = l1(xy[:,:,0:2], tg)
    # recurrent_loss
    recurrent_loss = th.sqrt(th.sum(th.square(policy.gru.weight_hh_l0)))

    loss = 1e-6*input_loss + 5*muscle_loss + 0.1*hidden_loss + 2*position_loss #+ 1e-5*recurrent_loss

    # Jon's proposed loss
    #position_loss = l1(xy[:,:,0:2],tg)
    #muscle_loss = th.mean(th.sum(th.square(all_force), dim=-1))
    #muscle_loss = th.mean(th.sum(all_force, dim=-1))
    #hidden_loss = th.mean(th.sum(th.square(all_hidden), dim=-1))
    #diff_loss =  th.mean(th.sum(th.square(th.diff(all_hidden, 1, dim=1)), dim=-1))

    #loss = position_loss + 5e-5*muscle_loss + 5e-5*hidden_loss + 3e-2*diff_loss
    #loss = position_loss + 1e-4*muscle_loss + 5e-5*hidden_loss + 1e-1*diff_loss
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    overall_losses.append(loss.item())
    hidden_losses.append(hidden_loss.item())
    muscle_losses.append(muscle_loss.item())

    # test the network
    # Run episode
    h = policy.init_hidden(batch_size=8)
    obs, info = env.reset(condition='test',catch_trial_perc=0,ff_coefficient=ff_coefficient,options={'batch_size':8})
    terminated = False

    # initial positions and targets
    xy = []
    tg = []
    vel = []
    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
      action, h = policy(obs,h)
      obs, _, terminated, _, info = env.step(action=action)

      #xy.append(info['states']['cartesian'][:, None, :])  # trajectories
      xy.append(info["states"]["fingertip"][:,None,:])  # trajectories
      tg.append(info["goal"][:, None, :])  # targets
      vel.append(info["states"]["cartesian"][:,None,2:]) # velocity

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=1)
    tg = th.cat(tg, axis=1)
    vel = th.cat(vel, axis=1)

    position_loss = l1(xy[:,:,0:2],tg)
    position_losses.append(position_loss.item())

    angle_loss = np.mean(calculate_angles_between_vectors(th.detach(vel), th.detach(tg), th.detach(xy)))
    angle_losses.append(angle_loss.item())

    lat_loss, _, _ = calculate_lateral_deviation(th.detach(xy), th.detach(tg))
    lat_loss = np.mean(lat_loss)
    lat_losses.append(lat_loss.item())

    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean position loss: {}".format(batch, n_batch, sum(position_losses[-interval:])/interval))

  # Save model
  weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
  log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
  cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")

  # save model weights
  th.save(policy.state_dict(), weight_file)

  # save training history (log)
  with open(log_file, 'w') as file:
    #json.dump(position_losses, file)
    json.dump({'overall_loss':overall_losses,'muscle_loss':muscle_losses,'hidden_loss':hidden_losses,'position_loss':position_losses,'angle_loss':angle_losses,'lat_loss':lat_losses}, file)

  # save environment configuration dictionary
  cfg = env.get_save_config()
  with open(cfg_file, 'w') as file:
    json.dump(cfg,file)

  print("done.")

def test(cfg_file,weight_file,ff_coefficient=None):
  device = th.device("cpu")

  # load configuration
  with open(cfg_file,'r') as file:
    cfg = json.load(file)

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
  
  batch_size = 8
  # initialize batch
  obs, info = env.reset(condition ='test',catch_trial_perc=0,options={'batch_size':batch_size},ff_coefficient=ff_coefficient)

  h = policy.init_hidden(batch_size=batch_size)
  terminated = False

  # initial positions and targets
  xy = [info["states"]["fingertip"][:, None, :]]
  tg = [info["goal"][:, None, :]]
  vel = [info["states"]["cartesian"][:, None,2:]]
  all_actions = []
  all_muscles = []
  all_hidden = []

  # simulate whole episode
  while not terminated:  # will run until `max_ep_duration` is reached
    all_hidden.append(h[0,:,None,:])
    action, h = policy(obs, h)

    obs, reward, terminated, truncated, info = env.step(action=action)  
    xy.append(info["states"]["fingertip"][:,None,:])  # trajectories
    tg.append(info["goal"][:,None,:])  # targets
    vel.append(info["states"]["cartesian"][:,None,2:])
    all_actions.append(action[:, None, :])
    
    all_muscles.append(info['states']['muscle'][:,0,None,:])
    

  # concatenate into a (batch_size, n_timesteps, xy) tensor
  xy = th.detach(th.cat(xy, axis=1))
  tg = th.detach(th.cat(tg, axis=1))
  vel = th.detach(th.cat(vel, axis=1))
  all_hidden = th.detach(th.cat(all_hidden, axis=1))
  all_actions = th.detach(th.cat(all_actions, axis=1))
  all_muscles = th.detach(th.cat(all_muscles, axis=1))

  return xy, tg, all_hidden, all_muscles, vel



if __name__ == "__main__":
    ## training single network - use for debugging
    # model_num = int(sys.argv[1])
    # ff_coefficient = float(sys.argv[2])
    # phase = int(sys.argv[3])
    # directory_name = sys.argv[4]
    # train(model_num,ff_coefficient,phase,directory_name)

    trainall = int(sys.argv[1])

    if trainall:
      directory_name = sys.argv[2]

      iter_list = range(16)
      n_jobs = 16
      while len(iter_list) > 0:
          these_iters = iter_list[0:n_jobs]
          iter_list = iter_list[n_jobs:]
          # pretraining the network using ADAM
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,0,condition='growing_up',directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,1,condition='train',directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,2,condition='train',directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,0,3,condition='train',directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,8,4,condition='train',directory_name=directory_name) 
                                                     for iteration in these_iters)
          
    else: ## training networks for each phase separately
      ff_coefficient = int(sys.argv[2])
      phase = int(sys.argv[3])
      condition = sys.argv[4]
      directory_name = sys.argv[5]

      iter_list = range(16)
      n_jobs = 16

      #train(1,ff_coefficient,phase,condition=condition,directory_name=directory_name)
      while len(iter_list) > 0:
          these_iters = iter_list[0:n_jobs]
          iter_list = iter_list[n_jobs:]
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,ff_coefficient,phase,condition=condition,directory_name=directory_name) 
                                                     for iteration in these_iters)

