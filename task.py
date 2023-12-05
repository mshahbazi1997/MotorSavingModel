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
            seed: int | None = None, 
            ff_coefficient: float = 0., 
            condition: str = 'train',
            catch_trial_perc: float = 50,
            is_channel: bool = False,
            calc_endpoint_force: bool = False,
            K: float = 1,
            B: float = -1,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.1, 0.3),
            pert_force_range: Union[list, tuple, np.ndarray] = (1, 2), # force in newton
            pert_time_range: Union[list, tuple, np.ndarray] = (0.1, 0.7), # time in seconds
            pert_dur_range: Union[list, tuple, np.ndarray] = (0.2, 0.3), # duration in seconds
            pert_prob: float = 0.0,
            options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

    self._set_generator(seed)

    self.calc_endpoint_force = calc_endpoint_force

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
    deterministic: bool = options.get('deterministic', False)


    self.batch_size = batch_size 
  
    self.catch_trial_perc = catch_trial_perc
    self.ff_coefficient = ff_coefficient
    self.go_cue_range = go_cue_range # in seconds

    # perturbation parameters
    self.pert_force_range = pert_force_range
    self.pert_time_range = pert_time_range
    self.pert_dur_range = pert_dur_range
    self.pert_prob = pert_prob

    # channel parameters
    self.is_channel = is_channel
    self.K = K
    self.B = B
    
    if (condition=='train'): # train net to reach to random targets

      joint_state = None

      goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      go_cue_time = np.random.uniform(self.go_cue_range[0],self.go_cue_range[1],batch_size)
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

      # Not sure about self.differentiable part TODO
      self.goal = goal if self.differentiable else self.detach(goal)

      # specify go cue time
      #go_cue_time = np.tile((self.go_cue_range[0]+self.go_cue_range[1])/2,batch_size)
      go_cue_time = np.tile(0.1,batch_size)
      self.go_cue_time = go_cue_time
      
    self.effector.reset(options={"batch_size": batch_size,"joint_state": joint_state})

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
    self.go_cue = th.zeros((batch_size,1)).to(self.device)


    obs = self.get_obs(deterministic=deterministic)

    # initial states
    self.init = self.states['fingertip']
    
    endpoint_load = th.zeros((batch_size,2)).to(self.device)
    self.endpoint_load = endpoint_load

    # perturbation....
    pert_trial = np.zeros(batch_size, dtype='float32')
    p = int(np.floor(batch_size * self.pert_prob / 100))
    pert_trial[np.random.permutation(pert_trial.size)[:p]] = 1.
    self.pert_trial = pert_trial

    # if perturbation trial, set endpoint load to perturbation force, if not set to zero
    pert_dur = np.random.uniform(self.pert_dur_range[0], self.pert_dur_range[1], batch_size).astype('float32')
    pert_time = np.random.uniform(self.pert_time_range[0],self.pert_time_range[1],batch_size).astype('float32')
    pert_interval = np.concatenate([pert_time[:,None],pert_time[:,None]+pert_dur[:,None]],axis=1)

    self.pert_interval = th.from_numpy(pert_interval).to(self.device)

    pert_dir = np.concatenate([np.random.uniform(-1,1,(batch_size,1)),np.random.uniform(-1,1,(batch_size,1))],axis=1).astype('float32')
    pert_dir = pert_dir / np.linalg.norm(pert_dir,axis=1,keepdims=True)
    pert_force = np.random.uniform(self.pert_force_range[0],self.pert_force_range[1],batch_size).astype('float32')
    
    self.pert_force = th.from_numpy(pert_force[:,None]*pert_dir).to(self.device)

    self.pert_interval[self.pert_trial==0] = self.max_ep_duration
    self.pert_force[self.pert_trial==0] = 0

    self.endpoint_force = th.zeros((batch_size,2)).to(self.device)
    
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
      "action": action,
      "noisy action": action,  # no noise here so it is the same
      "goal": self.goal,
      }
    return obs, info

  def step(self, action, deterministic: bool = False, **kwargs):
    self.elapsed += self.dt

    if deterministic is False:
      noisy_action = self.apply_noise(action, noise=self.action_noise)
    else:
      noisy_action = action
    
    self.effector.step(noisy_action,endpoint_load=self.endpoint_load) # **kwargs

    # Calculate endpoiont_load
    vel = self.states["cartesian"][:,2:]

    self.goal = self.goal.clone()
    self.init = self.init.clone()

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
      self.endpoint_load = F

      mask = self.elapsed < self.go_cue_time
      self.endpoint_load[mask] = 0

    elif self.pert_prob > 0:
      
      self.endpoint_load = th.zeros((self.batch_size,2)).to(self.device)
      mask = (self.elapsed > self.pert_interval[:,0]) & (self.elapsed < self.pert_interval[:,1])
      self.endpoint_load[mask] = self.pert_force[mask]


    else:
      FF_matvel = th.tensor([[0, 1], [-1, 0]], dtype=th.float32)
      # set endpoint load to zero before go cue
      self.endpoint_load = self.ff_coefficient * (vel@FF_matvel.T)

      mask = self.elapsed < self.go_cue_time
      self.endpoint_load[mask] = 0

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

      jacobian = th.stack([th.stack([jacobian_11, jacobian_12], dim=-1),
                            th.stack([jacobian_21, jacobian_22], dim=-1)], dim=-2)



      endpoint_force = th.zeros((self.batch_size,2)).to(self.device)

      forces = self.states['muscle'][:,self.muscle.state_name.index('force'):self.muscle.state_name.index('force')+1,:] # (batch_size,1,n_muscles)
      moments = self.states["geometry"][:, 2:, :]

      torque = - th.sum(forces * moments, dim=-1)


      for i in range(self.batch_size):
        jacobian = th.tensor([[jacobian_11[i],jacobian_12[i]],[jacobian_21[i],jacobian_22[i]]])
        
        endpoint_force[i] = torque[i] @ th.inverse(jacobian)
      self.endpoint_force = endpoint_force
    else:
      self.endpoint_force = th.zeros((self.batch_size,2)).to(self.device)


    
    

    # specify go cue time
    mask = self.elapsed >= (self.go_cue_time + (self.vision_delay-1) * self.dt)
    self.go_cue[mask] = 1

    obs = self.get_obs(action=noisy_action)
    reward = None
    truncated = False
    terminated = bool(self.elapsed >= self.max_ep_duration)
    info = {
      "states": self.states,
      "endpoint_load": self.endpoint_load,
      "endpoint_force": self.endpoint_force,
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
      self.go_cue, # sepcify go cue as an input to the network
      ]
    obs = th.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, noise=self.obs_noise)
    return obs