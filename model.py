import os
import sys 
from utils import create_directory
import motornet as mn
from task import CentreOutFF
from policy import Policy
import torch as th
import numpy as np
import json
from joblib import Parallel, delayed
from pathlib import Path



def _train(model_num,ff_coefficient,phase,directory_name=None):
  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  device = th.device("cpu")

  # Define task and the effector
  effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.ReluMuscle()) 

  if phase==1:
    #hidden_noise         = 1e-3
    action_noise         = 1e-4
    proprioception_noise = 1e-3
    vision_noise         = 1e-4
    vision_delay         = 0.05
    proprioception_delay = 0.02
    env = CentreOutFF(effector=effector,max_ep_duration=1,action_noise=action_noise,
                      proprioception_noise=proprioception_noise,vision_noise=vision_noise,
                      proprioception_delay=proprioception_delay,
                      vision_delay=vision_delay)
    # Define network
    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
  else:
    # load config and weights from the previous phase
    weight_file = [directory for directory in Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_weights') if directory.is_file()][0]
    cfg_file = [directory for directory in Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_cfg.json') if directory.is_file()][0]

    # load configuration
    with open(cfg_file,'r') as file:
      cfg = json.load(file)

    # effector
    muscle_name = cfg['effector']['muscle']['name']
    timestep = cfg['effector']['dt']
    muscle = getattr(mn.muscle,muscle_name)()
    effector = mn.effector.RigidTendonArm26(muscle=muscle,timestep=timestep)

    # delay
    proprioception_delay = cfg['proprioception_delay']*cfg['dt']
    vision_delay = cfg['vision_delay']*cfg['dt']

    # noise
    action_noise = cfg['action_noise'][0]
    proprioception_noise = cfg['proprioception_noise'][0]
    vision_noise = cfg['vision_noise'][0]

    # initialize environment
    max_ep_duration = cfg['max_ep_duration']
    env = CentreOutFF(effector=effector,max_ep_duration=max_ep_duration,name=cfg['name'],
                      action_noise=action_noise,proprioception_noise=proprioception_noise,
                      vision_noise=vision_noise,proprioception_delay=proprioception_delay,
                      vision_delay=vision_delay)

    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
    policy.load_state_dict(th.load(weight_file))

  
  optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)

  # Define Loss function
  def l1(x, y):
    """L1 loss"""
    return th.mean(th.sum(th.abs(x - y), dim=-1))

  # Train network
  batch_size = 32
  n_batch = 6000
  losses = []
  interval = 250

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
    muscle_loss = 0.1 * th.mean(th.sum(th.square(all_muscle), dim=-1))
    velocity_loss = 0.1 * th.mean(th.sum(th.abs(xy[:,:,2:]), dim=-1))
    input_loss = 1e-4 * th.sum(th.square(policy.gru.weight_ih_l0))
    recurrent_loss = 1e-4 * th.sum(th.square(policy.gru.weight_hh_l0))

    loss = cartesian_loss + muscle_loss + velocity_loss + input_loss + recurrent_loss
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    losses.append(loss.item())

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
    json.dump(losses, file)

  # save environment configuration dictionary
  cfg = env.get_save_config()
  with open(cfg_file, 'w') as file:
    json.dump(cfg, file)

  print("done.")

if __name__ == "__main__":
    model_num = int(sys.argv[1])
    ff_coefficient = float(sys.argv[1])
    phase = int(sys.argv[2])
    directory_name = sys.argv[3]
    #_train(model_num=model_num,ff_coefficient=ff_coefficient)

    iter_list = range(16)
    n_jobs = 8
    while len(iter_list) > 0:
        these_iters = iter_list[0:n_jobs]
        iter_list = iter_list[n_jobs:]
        result = Parallel(n_jobs=len(these_iters))(delayed(_train)(iteration,ff_coefficient,phase,directory_name=directory_name) for iteration in these_iters)

