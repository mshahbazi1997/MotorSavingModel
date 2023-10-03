import os
import datetime
import matplotlib.pyplot as plt
import motornet as mn
import numpy as np


def create_directory(directory_name=None):
    if directory_name is None:
        directory_name = datetime.datetime.now().date().isoformat()

    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Create the full directory path
    directory_path = os.path.join(home_directory, "Documents", "Data", directory_name)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Return the created directory's name (whether it was newly created or already existed)
    return directory_path

# average over non-overlapping windows size w;
# throws away tail of x that doesn't divide evenly into w
def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def plot_training_log(log):
    ax = plt.subplot(1,1,1)

    ax.semilogy(log)

    ax.set_ylabel("Loss")
    ax.set_xlabel("Batch #")
    return ax


def plot_simulations(xy, target_xy):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    plt.figure(figsize=(5,3))

    plt.subplot(1,1,1)
    plt.ylim([0.3, 0.65])
    plt.xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time


    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)

    #plt.subplot(1,2,2)
    #plt.ylim([-0.1, 0.1])
    #plt.xlim([-0.1, 0.1])
    #plotor(axis=plt.gca(), cart_results=xy - target_xy)
    #plt.axhline(0, c="grey")
    #plt.axvline(0, c="grey")
    #plt.xlabel("X distance to target")
    #plt.ylabel("Y distance to target")
    plt.show()

def load_env(task,cfg=None):

    if cfg is None:

        name = 'env'

        action_noise         = 1e-4
        proprioception_noise = 1e-3
        vision_noise         = 1e-4
        vision_delay         = 0.05
        proprioception_delay = 0.02

        # Define task and the effector
        effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.ReluMuscle())

        max_ep_duration = 1
    else:
        name = cfg['name']
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


    env = task(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    return env




        