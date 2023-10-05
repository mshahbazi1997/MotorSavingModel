import os
import sys 
from utils import create_directory, load_env
import motornet as mn
from task import CentreOutFF
from policy import Policy
import torch as th
import numpy as np
import json
from joblib import Parallel, delayed
from pathlib import Path

def train(model_num,ff_coefficient,phase,directory_name=None):
  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  device = th.device("cpu")

  if phase>1:
    # load config and weights from the previous phase
    weight_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_weights'))[0]
    cfg_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_cfg.json'))[0]

    # load configuration
    with open(cfg_file,'r') as file:
      cfg = json.load(file)

    # environment and network
    env = load_env(CentreOutFF,cfg)
    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
    policy.load_state_dict(th.load(weight_file))

  else:
    # environment and network
    env = load_env(CentreOutFF)    
    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
  
  optimizer = th.optim.Adam(policy.parameters(), lr=0.001) # 10**-3

  # Define Loss function
  def l1(x, y):
    """L1 loss"""
    return th.mean(th.sum(th.abs(x - y), dim=-1))

  # Train network
  batch_size = 65
  n_batch = 50000
  losses = []
  position_loss = []
  interval = 1000

  for batch in range(n_batch):

    # check if you want to load a model TODO
    h = policy.init_hidden(batch_size=batch_size)

    obs, info = env.reset(condition = "train",ff_coefficient=ff_coefficient, options={'batch_size':batch_size})
    terminated = False

    # initial positions and targets
    xy = [info['states']['cartesian'][:, None, :]]
    tg = [info["goal"][:, None, :]]
    all_actions = []
    all_muscle = []
    all_hidden = []

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
      action, h = policy(obs,h)
      obs, reward, terminated, truncated, info = env.step(action=action)

      xy.append(info['states']['cartesian'][:, None, :])  # trajectories
      tg.append(info["goal"][:, None, :])  # targets
      all_actions.append(action[:, None, :])
      all_muscle.append(info['states']['muscle'][:,0,None,:])
      all_hidden.append(h[0,:,None,:])

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=1)
    tg = th.cat(tg, axis=1)
    all_hidden = th.cat(all_hidden, axis=1)
    all_actions = th.cat(all_actions, axis=1)
    all_muscle = th.cat(all_muscle, axis=1)

    # calculate losses
    cartesian_loss = l1(xy[:,:,0:2], tg)
    action_loss = 1e-5 * th.sum(th.abs(all_actions))
    hidden_loss = 1e-6 * th.sum(th.abs(all_hidden))
    hidden_diff_loss = 1e-7 * th.sum(th.abs(th.diff(all_hidden, dim=1)))
    #muscle_loss = 0.1 * th.mean(th.sum(th.square(all_muscle), dim=-1))

    loss = cartesian_loss + action_loss + hidden_loss + hidden_diff_loss
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    losses.append(loss.item())
    position_loss.append(cartesian_loss.item())

    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean policy loss: {}".format(batch, n_batch, sum(losses[-interval:])/interval))

  # Save model
  weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
  log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
  cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")


  # save model weights
  th.save(policy.state_dict(), weight_file)

  # save training history (log)
  with open(log_file, 'w') as file:
    #json.dump(losses, file)
    json.dump({'losses':losses,'position_loss':position_loss}, file)

  # save environment configuration dictionary
  cfg = env.get_save_config()
  with open(cfg_file, 'w') as file:
    json.dump(cfg, file)

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
  policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
  policy.load_state_dict(th.load(weight_file))

  batch_size = 8
  # initialize batch
  obs, info = env.reset(condition ="test",catch_trial_perc=0,options={'batch_size':batch_size},ff_coefficient=ff_coefficient)

  h = policy.init_hidden(batch_size=batch_size)
  terminated = False

  # initial positions and targets
  xy = [info["states"]["fingertip"][:, None, :]]
  tg = [info["goal"][:, None, :]]

  # simulate whole episode
  while not terminated:  # will run until `max_ep_duration` is reached
    action, h = policy(obs, h)

    obs, reward, terminated, truncated, info = env.step(action=action)  
    xy.append(info["states"]["fingertip"][:,None,:])  # trajectories
    tg.append(info["goal"][:,None,:])  # targets

  # concatenate into a (batch_size, n_timesteps, xy) tensor
  xy = th.detach(th.cat(xy, axis=1))
  tg = th.detach(th.cat(tg, axis=1))

  return xy, tg


if __name__ == "__main__":
    model_num = int(sys.argv[1])
    ff_coefficient = float(sys.argv[2])
    phase = int(sys.argv[3])
    directory_name = sys.argv[4]
    train(model_num,ff_coefficient,phase,directory_name)


    #ff_coefficient = float(sys.argv[1])
    #phase = int(sys.argv[2])
    #directory_name = sys.argv[3]

    # iter_list = range(16)
    # n_jobs = 16
    # while len(iter_list) > 0:
    #     these_iters = iter_list[0:n_jobs]
    #     iter_list = iter_list[n_jobs:]
    #     result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,ff_coefficient,phase,directory_name=directory_name) for iteration in these_iters)

