import os
import datetime
import motornet as mn
import numpy as np
from scipy.optimize import minimize

def create_directory(directory_name=None):
    if directory_name is None:
        directory_name = datetime.datetime.now().date().isoformat()

    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Create the full directory path
    directory_path = os.path.join(home_directory, "Documents", "Data","MotorNet", directory_name)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    #else:
        #print(f"Directory '{directory_path}' already exists.")

    # Return the created directory's name (whether it was newly created or already existed)
    return directory_path

def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def load_env(task,cfg=None,dT=None):

    if cfg is None:

        name = 'env'

        action_noise         = 1e-4
        proprioception_noise = 1e-3 # 1e-3
        vision_noise         = 1e-3 # 1e-4
        vision_delay         = 0.07
        proprioception_delay = 0.02

        # Define task and the effector
        effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

        max_ep_duration = 1
    else:
        name = cfg['name']
        # effector
        muscle_name = cfg['effector']['muscle']['name']
        #timestep = cfg['effector']['dt']
        if dT is None:
            timestep = cfg['dt']
        else:
            timestep = dT
        cfg['dt'] = timestep
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


    env = task(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    return env


def load_policy(env,modular=0,freeze_output_layer=False, freeze_input_layer=False, freeze_bias_hidden=False, freeze_h0=False, weight_file=None):
    import torch as th
    device = th.device("cpu")
    if modular:
        from motornet.policy import ModularPolicyGRU


        # # PMd, M1, S1, Spinal
        # vision_mask = [0.2, 0.02, 0, 0]
        # proprio_mask = [0, 0, 0, 0.5]
        # task_mask = [0.2, 0.02, 0, 0]
        # connectivity_mask = np.array([[1, 0.2, 0.02, 0],
        #                                 [0.2, 1, 0.2, 0.02],
        #                                 [0.02, 0.02, 1, 0.2],
        #                                 [0, 0.2, 0.02, 1]])
        # connectivity_delay = np.array([[0, 1, 1, 1, 1],
        #                                 [1, 0, 1, 1, 1],
        #                                 [1, 1, 0, 1, 1],
        #                                 [1, 1, 1, 0, 1],
        #                                 [1, 1, 1, 1, 0]])
        # connectivity_delay = np.zeros_like(connectivity_mask)
        # output_mask = [0, 0, 0, 0.5]
        # module_sizes = [128, 128, 128, 32]
        # spectral_scaling = 1.1
        
        # PFC, PMd, M1, S1, PPC, Spinal
        proportion_excitatory = None#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        vision_mask = [0, 0, 0, 0, 1, 0]
        proprio_mask = [0, 0, 0, 0, 0, 1]
        task_mask = [0, 0, 0, 0, 1, 0]
        c1 = 0.2
        c2 = 0.05
        c3 = 0.01
        c0 = 0.5
        connectivity_mask = np.array([[c0, c1, c3, 0, c1, 0],
                                    [c1, c0, c1, c3, c1, 0],
                                    [c2, c1, c0, c1, c3, c3],
                                    [0, 0, c3, c0, c3, c1],
                                    [c1, c1, c3, c1, c0, 0],
                                    [0, 0, c1, 0, 0, c0]])
        connectivity_mask[connectivity_mask > 1] = 1
        connectivity_delay = np.zeros_like(connectivity_mask)
        output_mask = [0, 0, 0, 0, 0, 1]
        module_sizes = [128, 128, 128, 128, 128, 16]
        spectral_scaling = 1.2

        # goal, vision, proprioception, go_cue
        task_dim = np.array([0,1,16]) # goal, go_cue
        vision_dim = np.array([2,3])
        proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1

        policy = ModularPolicyGRU(env.observation_space.shape[0], module_sizes, env.n_muscles, 
                                  vision_dim=vision_dim, proprio_dim=proprio_dim, task_dim=task_dim, 
                                  vision_mask=vision_mask, proprio_mask=proprio_mask, task_mask=task_mask,
                                  connectivity_mask=connectivity_mask, output_mask=output_mask, connectivity_delay=connectivity_delay,
                                  spectral_scaling=spectral_scaling, device=device, activation='tanh')
    else:
        #num_hidden = 128
        num_hidden = 128
        from policy import Policy
        policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, 
                        freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer, 
                        freeze_bias_hidden=freeze_bias_hidden, freeze_h0=freeze_h0)
        
    if weight_file is not None:
        policy.load_state_dict(th.load(weight_file,map_location=device))
    return policy
        

def calculate_angles_between_vectors(vel, tg, xy):
    """
    Calculate angles between vectors X2 and X3.

    Parameters:
    - vel (numpy.ndarray): Velocity array.
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - angles (numpy.ndarray): An array of angles in degrees between vectors X2 and X3.
    """

    tg = np.array(tg)
    xy = np.array(xy)
    vel = np.array(vel)
    
    # Compute the magnitude of velocity and find the index to the maximum velocity
    vel_norm = np.linalg.norm(vel, axis=-1)
    idx = np.argmax(vel_norm, axis=1)

    # Calculate vectors X2 and X3
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]
    X3 = xy[np.arange(xy.shape[0]), idx, :]

    X2 = X2 - X1
    X3 = X3 - X1
    
    cross_product = np.cross(X3, X2)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)

    # Calculate the angles in degrees
    angles = sign*np.degrees(np.arccos(np.sum(X2 * X3, axis=1) / (1e-8+np.linalg.norm(X2, axis=1) * np.linalg.norm(X3, axis=1))))

    return angles

def calculate_lateral_deviation(xy, tg, vel=None):
    """
    Calculate the lateral deviation of trajectory xy from the line connecting X1 and X2.

    Parameters:
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - deviation (numpy.ndarray): An array of lateral deviations.
    """
    tg = np.array(tg)
    xy = np.array(xy)

    # Calculate vectors X2 and X1
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]

    # Calculate the vector representing the line connecting X1 to X2
    line_vector = X2 - X1
    line_vector2 = np.tile(line_vector[:,None,:],(1,xy.shape[1],1))

    # Calculate the vector representing the difference between xy and X1
    trajectory_vector = xy - X1[:,None,:]

    projection = np.sum(line_vector2 * trajectory_vector, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
    projection = line_vector2 * projection[:,:,np.newaxis]

    lateral_dev = np.linalg.norm(trajectory_vector - projection,axis=2)

    idx = np.argmax(lateral_dev,axis=1)

    max_laterl_dev = lateral_dev[np.arange(idx.shape[0]), idx]

    init = projection[np.arange(idx.shape[0]),idx,:]
    init = init+X1

    endp = xy[np.arange(idx.shape[0]),idx,:]


    cross_product = np.cross(endp-X1, X2-X1)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)


    opt={'lateral_dev':np.mean(lateral_dev,axis=-1),
         'max_lateral_dev':max_laterl_dev,
         'lateral_vel':None}
    # speed 
    if vel is not None:
        vel = np.array(vel)
        projection = np.sum(line_vector2 * vel, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
        projection = line_vector2 * projection[:,:,np.newaxis]
        lateral_vel = np.linalg.norm(vel - projection,axis=2)
        opt['lateral_vel'] = np.mean(lateral_vel,axis=-1)




    return sign*max_laterl_dev, init, endp, opt

def optimize_channel(cfg_file,weight_file):

    def lat_loss(theta):
        from model import test
        data = test(cfg_file,weight_file,is_channel=True,K=theta[0],B=theta[1])
        _, _, _, opt = calculate_lateral_deviation(data['xy'], data['tg'], data['vel'])
        return np.mean(opt['max_lateral_dev'])

    # find K and B such that max lateral deviation is minimized...
    loss_before = lat_loss([0,0])

    theta0 = [180,-2]
    theta = minimize(lat_loss,theta0,method='Nelder-Mead',options={'maxiter':10000,'disp':False})
    loss_after = lat_loss(theta.x)

    print(f'loss before: {loss_before}')
    print(f'loss after: {loss_after}')

    return theta.x

