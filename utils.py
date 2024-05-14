import os
import motornet as mn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from copy import deepcopy


base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def load_env(task,cfg=None,dT=None):
    # also get K and B
    if cfg is None:

        name = 'env'

        action_noise         = 1e-4
        proprioception_noise = 1e-3
        vision_noise         = 1e-4
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


def load_policy(n_input,n_output,weight_file=None,phase='growing_up',freeze_output_layer=False,freeze_input_layer=False,n_hidden=128):

    import torch as th
    device = th.device("cpu")
    
    from policy import Policy
    policy = Policy(n_input, n_hidden, n_output, device=device, 
                    freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer)
    
    if weight_file is not None:
        policy.load_state_dict(th.load(weight_file,map_location=device))

    if phase=='growing_up':
        optimizer = th.optim.Adam(policy.parameters(), lr=3e-3)
        scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.9999)
    else:
        optimizer = th.optim.SGD(policy.parameters(), lr=5e-3)
        scheduler = None
        
    return policy, optimizer, scheduler


def load_stuff(cfg_file,weight_file,phase='growing_up',freeze_output_layer=False, freeze_input_layer=False,n_hidden=128):
    # also get K and B
    import json
    from task import CentreOutFF

    # load configuration
    cfg = None
    if cfg_file is not None:
        cfg = json.load(open(cfg_file, 'r'))
    env = load_env(CentreOutFF, cfg)

    n_input = env.observation_space.shape[0]
    n_output = env.n_muscles

    # load policy
    policy, optimizer, scheduler = load_policy(n_input,n_output,weight_file=weight_file,phase=phase,
                                               freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer,n_hidden=n_hidden)

    return env, policy, optimizer, scheduler
        

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

def sweep_loss():
    loss_weights = np.array([1e+3,   # position
                                 1e+5,   # jerk
                                 1e-1,   # muscle
                                 1e-5,   # muscle_derivative
                                 3e-5,   # hidden 
                                 2e-2,   # hidden_derivative
                                 0])     # hidden_jerk
    md_w = np.logspace(start=-6,stop=-1,num=3)
    h_w = np.logspace(start=-5,stop=-2,num=3)
    hd_w = np.logspace(start=-3,stop=-1,num=3)

    mhh_w = np.array(list(product(md_w,h_w,hd_w)))

    iter_list = range(20)
    num_processes = len(iter_list)

    lw = [loss_weights.copy() for _ in range(len(mhh_w))]

    for idx,mhhw in enumerate(mhh_w):
        lw[idx][3] = mhhw[0]
        lw[idx][4] = mhhw[1]
        lw[idx][5] = mhhw[2]
    return lw


class modelLoss():
    def __init__(self):
        #self.n_param = 5
        pass
    def predict(self,theta=None):
        if theta is None:
            theta = self.theta
        #pred = theta[0]*np.exp(-theta[1]*self.x**2)+ theta[2]

        # pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**3) +\
        #       np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**2) +\
        #         np.exp(theta[4])*np.exp(-np.exp(theta[5])*self.x**1) + theta[6] 
        
        #pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**3) +\
        #        np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**1) + theta[4] 
        

        # pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**=3) +\
        #         np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**1) + theta[4] 
        pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**1) +\
            + theta[2] 


        return pred
    def fro(self,theta,loss,lam=None):
        pred = self.predict(theta)
        return np.linalg.norm(loss-pred)+lam*np.sum(np.abs(theta))
    def fit(self,loss,lam=None,theta0=None):
        if lam is None:
            lam = 0
        
        self.x = np.arange(len(loss))

        if theta0 is None:
            #theta0 = [-4.87692324e-05, -9.65588447e-06,  2.65996536e+00, -4.56851626e+00,2.18185688e+01]
            theta0 = [-1, -1,22]
        theta = minimize(self.fro,theta0,args=(loss,lam),method='Nelder-Mead',options={'maxiter':10000,'disp':False})
        self.theta = theta.x
        self.success = theta.success
    def find_x(self,loss_thresh):
        pred = self.predict()
        idx = np.where(pred<=loss_thresh)[0]
        return idx[0]
    def get_rate(self):
        return np.exp(self.theta[1])
    
def get_initial_loss(loss):
    
    T = pd.DataFrame()

    # get initial loss
    data = {'NF1':[],'FF1':[],'NF2':[],'FF2':[]}
    for p in list(data.keys()):
        index=0
        if p=='NF1' or p=='NF2':
            index=-1
        data[p] = list(np.array(loss[p])[:,index])

    T = create_dataframe(data)
    T['feature'] = 'init'
    
    return T

def get_rate(loss,w=10,check_fit=False):
    loss2 = deepcopy(loss)
    if w>1:
        for phase in loss2.keys():
            loss2[phase] = [window_average(np.array(l), w) for l in loss2[phase]]
    

    T = pd.DataFrame()

    # Fit data
    data = {'FF1':[],'FF2':[]} # this will contain the rate
    pred = {'FF1':[],'FF2':[]} # just for checking the fit
    
    for m in range(len(loss2['FF1'])):
        for _,phase in enumerate(data.keys()):
            l = loss2[phase][m]

            model = modelLoss()

            theta0=[np.log(l[0]),np.log(0.004),l[-1]]
            model.fit(l,lam=0.0,theta0=theta0)

            pred[phase].append(model.predict())
            data[phase].append(model.get_rate())
    
    T = create_dataframe(data)
    T['feature'] = 'rate'

    # Check the fits
    if check_fit:
        _,ax = plt.subplots(1,2,figsize=(6,5))
        ax[0].plot(np.mean(loss2['FF1'],axis=0),linestyle='-',color='b',label='data')
        ax[0].plot(np.mean(pred['FF1'],axis=0),linestyle='--',color='r',label='pred')
        ax[0].legend()

        ax[1].plot(np.mean(loss2['FF2'],axis=0),linestyle='-',color='b',label='data')
        ax[1].plot(np.mean(pred['FF2'],axis=0),linestyle='--',color='r',label='pred')
        ax[1].legend()
        plt.show()

    return T


def create_dataframe(idx):
    data = []
    for p in list(idx.keys()):
        val = idx[p]
        data.extend([
            {'mn': i + 1, 'phase': p, 'value': v}
            for i, v in enumerate(val)
        ])
    return pd.DataFrame(data)