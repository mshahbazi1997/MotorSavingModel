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


def _train(model_num,ff_coefficient):
  output_folder = create_directory()
  device = th.device("cpu")

  # Define task and the effector
  effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.ReluMuscle()) 
  env = CentreOutFF(effector=effector,max_ep_duration=1.,name='env',)

  # Define network
  policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
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
    # initialize batch

    # check if you want to load a model TODO
    h = policy.init_hidden(batch_size=batch_size)

    obs, info = env.reset(condition = "train",ff_coefficient=ff_coefficient, options={'batch_size':batch_size})
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
      action, h = policy(obs, h)
      obs, reward, terminated, truncated, info = env.step(action=action)

      xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
      tg.append(info["goal"][:, None, :])  # targets

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=1)
    tg = th.cat(tg, axis=1)
    loss = l1(xy, tg)  # L1 loss on position
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    losses.append(loss.item())

    if (batch % interval == 0) and (batch != 0):
      print("Batch {}/{} Done, mean policy loss: {}".format(batch, n_batch, sum(losses[-interval:])/interval))

  # Save model
  weight_file = os.path.join(output_folder, f"model{model_num}_weights")
  log_file = os.path.join(output_folder, f"model{model_num}_log.json")
  cfg_file = os.path.join(output_folder, f"model{model_num}_cfg.json")


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
    ff_coefficient = float(sys.argv[2])
    _train(model_num=model_num,ff_coefficient=ff_coefficient)
    # iter_list = range(4)
    # n_jobs = 2
    # while len(iter_list) > 0:
    #     these_iters = iter_list[0:n_jobs]
    #     iter_list = iter_list[n_jobs:]
    #     result = Parallel(n_jobs=len(these_iters))(delayed(_train)(iteration,8) for iteration in these_iters)

