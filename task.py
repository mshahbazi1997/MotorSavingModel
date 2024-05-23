import motornet as mn
import torch as th
import numpy as np
from typing import Any
from typing import Union

go_time = 0.44

class CentreOutFF(mn.environment.Environment):
  """A reach to a random target from a random starting position."""

  def __init__(self, *args, **kwargs):
    # pass everything as-is to the parent Environment class
    super().__init__(*args, **kwargs)
    self.__name__ = "CentreOutFF"
    # check if we have K and B in kwargs
    self.K = kwargs.get('K', 150)
    self.B = kwargs.get('B', 0.5)


  def reset(self, *,
            seed: int | None = None,
            ff_coefficient: float = 0.,
            condition: str = 'train',
            catch_trial_perc: float = 50,
            go_cue_random = None,
            is_channel: bool = False,
            calc_endpoint_force: bool = False,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.1, 0.3),
            options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

    self._set_generator(seed)

    

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)

    self.calc_endpoint_force = calc_endpoint_force
    self.batch_size = batch_size 
    self.catch_trial_perc = catch_trial_perc
    self.ff_coefficient = ff_coefficient
    self.go_cue_range = go_cue_range # in seconds
    self.is_channel = is_channel


    if (condition=='train'): # train net to reach to random targets

      joint_state = None

      goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      if go_cue_random is None:
        go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
      else:
        if go_cue_random:
          go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
        else:
          go_cue_time = np.tile(go_time,batch_size)

      self.go_cue_time = go_cue_time

    elif (condition=='test'): # centre-out reaches to each target

      angle_set = np.deg2rad(np.arange(0,360,45)) # 8 directions
      reps        = int(np.ceil(batch_size / len(angle_set)))
      angle       = np.tile(angle_set, reps=reps)
      batch_size  = reps * len(angle_set)

      reaching_distance = 0.10
      lb = np.array(self.effector.pos_lower_bound)
      ub = np.array(self.effector.pos_upper_bound)
      start_position = lb + (ub - lb) / 2
      
      start_position = np.array([1.047, 1.570])
      start_position = start_position.reshape(1,-1)
      start_jpv = th.from_numpy(np.concatenate([start_position, np.zeros_like(start_position)], axis=1)) # joint position and velocity
      start_cpv = self.joint2cartesian(start_jpv).numpy()
      end_cp = reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)

      goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
      goal_states = goal_states[:,:2]
      goal_states = goal_states.astype(np.float32)

      joint_state = th.from_numpy(np.tile(start_jpv,(batch_size,1)))
      goal = th.from_numpy(goal_states)
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      if go_cue_random is None:
        go_cue_time = np.tile(go_time,batch_size)
      else:
        if go_cue_random:
          go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
        else:
          go_cue_time = np.tile(go_time,batch_size)
      self.go_cue_time = go_cue_time
      
    self.effector.reset(options={"batch_size": batch_size,"joint_state": joint_state})

    self.elapsed = 0.
    action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)
  
    self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
    self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
    self.obs_buffer["action"] = [action] * self.action_frame_stacking
    
    # specify catch trials
    catch_trial = np.zeros(batch_size, dtype='float32')
    p = int(np.floor(batch_size * self.catch_trial_perc / 100))
    catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
    self.catch_trial = catch_trial

    # specify go cue time
    self.go_cue_time[self.catch_trial==1] = self.max_ep_duration
    self.go_cue = th.zeros((batch_size,1)).to(self.device)
    self.init = self.states['fingertip']

    obs = self.get_obs(deterministic=deterministic)

    
    
    self.endpoint_load = th.zeros((batch_size,2)).to(self.device)
    self.endpoint_force = th.zeros((batch_size,2)).to(self.device)
    
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
      "action": action,
      "noisy action": action,  # no noise here so it is the same
      "goal": self.goal * self.go_cue + self.init * (1-self.go_cue), # target
      }
    return obs, info


  def step(self, action, deterministic: bool = False):
    self.elapsed += self.dt

    if deterministic is False:
      noisy_action = self.apply_noise(action, noise=self.action_noise)
    else:
      noisy_action = action
    
    self.effector.step(noisy_action,endpoint_load=self.endpoint_load)


    # calculate endpoint force (External force)
    self.endpoint_load = get_endpoint_load(self)
    mask = self.elapsed < (self.go_cue_time + (self.vision_delay) * self.dt)
    self.endpoint_load[mask] = 0

    # calculate endpoint force (Internal force)
    self.endpoint_force = get_endpoint_force(self)

    # specify go cue time
    #mask = self.elapsed >= (self.go_cue_time + (self.vision_delay-1) * self.dt)
    mask = self.elapsed > (self.go_cue_time + (self.vision_delay) * self.dt)
    self.go_cue[mask] = 1

    
    obs = self.get_obs(action=noisy_action)

    terminated = bool(self.elapsed >= self.max_ep_duration)
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
      "action": action,
      "noisy action": noisy_action,
      "goal": self.goal * self.go_cue + self.init * (1-self.go_cue),
      }
    
    return obs, terminated, info

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
      self.obs_buffer["vision"][0],  # oldest element
      self.obs_buffer["proprioception"][0],   # oldest element
      self.goal, # goal #self.init, # initial position
      self.go_cue, # sepcify go cue as an input to the network
      ]
    obs = th.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, noise=self.obs_noise)
    return obs
  

def get_endpoint_force(self):
  """Internal force
  """
  endpoint_force = th.zeros((self.batch_size, 2)).to(self.device)
  if self.calc_endpoint_force:
      L1 = self.skeleton.L1
      L2 = self.skeleton.L2

      pos0, pos1 = self.states['joint'][:,0], self.states['joint'][:,1]
      pos_sum = pos0 + pos1
      c1 = th.cos(pos0)
      c12 = th.cos(pos_sum)
      s1 = th.sin(pos0)
      s12 = th.sin(pos_sum)

      jacobian_11 = -L1*s1 - L2*s12
      jacobian_12 = -L2*s12
      jacobian_21 = L1*c1 + L2*c12
      jacobian_22 = L2*c12


      forces = self.states['muscle'][:, self.muscle.state_name.index('force'):self.muscle.state_name.index('force')+1, :]
      moments = self.states["geometry"][:, 2:, :]

      torque = -th.sum(forces * moments, dim=-1)

      for i in range(self.batch_size):
          jacobian_i = th.tensor([[jacobian_11[i], jacobian_12[i]], [jacobian_21[i], jacobian_22[i]]])

          endpoint_force[i] = torque[i] @ th.inverse(jacobian_i)

      return endpoint_force
  else:
      return endpoint_force

def get_endpoint_load(self):
  """External force
  """
  # Calculate endpoiont_load
  vel = self.states["cartesian"][:,2:]

  # TODO
  self.goal = self.goal.clone()
  self.init = self.init.clone()

  endpoint_load = th.zeros((self.batch_size,2)).to(self.device)

  if self.is_channel:

    X2 = self.goal
    X1 = self.init

    # vector that connect initial position to the target
    line_vector = X2 - X1

    xy = self.states["cartesian"][:,2:]
    xy = xy - X1

    projection = th.sum(line_vector * xy, axis=-1)/th.sum(line_vector * line_vector, axis=-1)
    projection = line_vector * projection[:,None]

    err = xy - projection

    projection = th.sum(line_vector * vel, axis=-1)/th.sum(line_vector * line_vector, axis=-1)
    projection = line_vector * projection[:,None]
    err_d = vel - projection
    
    F = -1*(self.B*err+self.K*err_d)
    endpoint_load = F
  else:
    FF_matvel = th.tensor([[0, 1], [-1, 0]], dtype=th.float32)
    endpoint_load = self.ff_coefficient * (vel@FF_matvel.T)
  return endpoint_load
