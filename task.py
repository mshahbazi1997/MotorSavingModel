import motornet as mn
import torch as th
import numpy as np
from typing import Any
from typing import Union


class CentreOutFF(mn.environment.Environment):
  """A reach to a random target from a random starting position."""

  def __init__(self, *args, **kwargs):
    # pass everything as-is to the parent Environment class
    super().__init__(*args, **kwargs)
    self.__name__ = "CentreOutFF"

  def reset(self, *, 
            seed: int = None, 
            ff_coefficient: float = 0., 
            condition: str = "train",
            catch_trial_perc: float = 0,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.0, 0.0),
            options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:

    self._set_generator(seed)


    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)
  
    self.catch_trial_perc = catch_trial_perc
    self.ff_coefficient = ff_coefficient

    #go_cue_range = np.array(go_cue_range) / self.dt
    self.go_cue_range = go_cue_range # in seconds
    self.delay_range = self.go_cue_range
    
    if batch_size is None:
        batch_size = 1
    
    if (condition=="train"): # train net to reach to random targets

      joint_state = None

      goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
      self.go_cue_time = go_cue_time

    elif (condition=="test"): # centre-out reaches to each target

      angle_set = np.deg2rad(np.arange(0,360,45)) # 8 directions
      reps        = int(np.ceil(batch_size / len(angle_set)))
      angle       = np.tile(angle_set, reps=reps)
      batch_size  = reps * len(angle_set)

      # In the reset, i will check if the trial is a catch trial or not
      catch_trial = np.zeros(batch_size, dtype='float32')

      reaching_distance = 0.1
      
      lb = np.array(self.effector.pos_lower_bound)
      ub = np.array(self.effector.pos_upper_bound)
      start_position = lb + (ub - lb) / 2
      start_position = start_position.reshape(1,-1)
      start_jpv = th.from_numpy(np.concatenate([start_position, np.zeros_like(start_position)], axis=1)) # joint position and velocity
      start_cpv = self.joint2cartesian(start_jpv).numpy()
      end_cp = reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)

      goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
      goal_states = goal_states[:,:2]
      goal_states = goal_states.astype(np.float32)

      joint_state = th.from_numpy(np.tile(start_jpv,(batch_size,1)))
      goal = th.from_numpy(goal_states)

      # Not sure about self.differentiable part TODO
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      go_cue_time = np.tile((self.go_cue_range[0]+self.go_cue_range[1])/2,batch_size)
      self.go_cue_time = go_cue_time
      
    self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

    self.elapsed = 0.
    action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
  
    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking

    # specify catch trials
    # In the reset, i will check if the trial is a catch trial or not
    catch_trial = np.zeros(batch_size, dtype='float32')
    p = int(np.floor(batch_size * self.catch_trial_perc / 100))
    catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
    self.catch_trial = catch_trial

    # if catch trial, set the go cue time to max_ep_duration
    # thus the network will not see the go-cue
    self.go_cue_time[self.catch_trial==1] = self.max_ep_duration
    #self.go_cue = th.zeros((batch_size,1)).to(self.device)
    self.go_cue = th.ones((batch_size,1)).to(self.device)

    obs = self.get_obs(deterministic=deterministic)

    # initial states
    self.init = self.states['fingertip']

    info = {
      "states": self.states,
      "action": action,
      "noisy action": action,  # no noise here so it is the same
      "goal": self.goal,
      }
    
    return obs, info

  def step(self, action, deterministic: bool = False):
    self.elapsed += self.dt

    if deterministic is False:
      noisy_action = self.apply_noise(action, noise=self.action_noise)
    else:
      noisy_action = action
    
    # Calculate endpoiont_load
    vel = self.effector.joint2cartesian(self.states['joint'])[:,2:]
    FF_matvel = th.tensor([[0, 1], [-1, 0]], dtype=th.float32)
    projection_x = th.sum(FF_matvel[0] * vel, dim=1, keepdim=True) 
    projection_y = th.sum(FF_matvel[1] * vel, dim=1, keepdim=True)
    projection = th.cat([projection_x, projection_y], dim=1)
    endpoint_load = self.ff_coefficient * projection

    # add endpoiont_load
    #self.effector.step(noisy_action,endpoiont_load=endpoint_load)
    self.effector.step(noisy_action)
    self.goal = self.goal.clone()
    self.init = self.init.clone()

    # specify go cue time
    #mask = self.elapsed >= (self.go_cue_time + self.vision_delay * self.dt)
    #self.go_cue[mask] = 1


    obs = self.get_obs(action=noisy_action)
    reward = None
    truncated = False
    terminated = bool(self.elapsed >= self.max_ep_duration)
    info = {
      "states": self.states,
      "action": action,
      "noisy action": noisy_action,
      "goal": self.goal * self.go_cue + self.init * (1-self.go_cue), # update the target depending on the go cue
      }
    return obs, reward, terminated, truncated, info

  def get_proprioception(self):
    mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
    mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
    prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
    return self.apply_noise(prop, self.proprioception_noise)

  def get_vision(self):
    vis = self.states["fingertip"]
    return self.apply_noise(vis, self.vision_noise)

  def get_obs(self, action=None, deterministic: bool = False):
    self.update_obs_buffer(action=action)

    obs_as_list = [
      self.goal,
      self.obs_buffer["vision"][0],  # oldest element
      self.obs_buffer["proprioception"][0],   # oldest element
      #self.go_cue, # sepcify go cue as an input to the network
      ]
    obs = th.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, noise=self.obs_noise)
    return obs