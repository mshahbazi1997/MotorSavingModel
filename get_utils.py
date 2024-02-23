import os
from pathlib import Path
from utils import load_stuff
from model import test
import numpy as np
import torch as th
import PcmPy as pcm
from PcmPy.matrix import indicator
from scipy.linalg import pinv

base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def get_dir(folder_name,model_name,phase,ff_coef,batch=None):
    """
    Get the directory of the model with the given parameters
        params:
            ff_coef: the coeffient by which we train the model, can be found in the phase dictionary
    """

    data_dir = os.path.join(base_dir,folder_name)
    cfg_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_cfg.json'))[0]
    loss_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_log.json'))[0]
    if batch is None:
        weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_weights'))[0]
    else:
        weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_batch={batch}_FFCoef={ff_coef}_weights'))[0]
    return weight_file, cfg_file, loss_file

def get_data(folder_name,model_name,phase={'NF1':[0]},ff_coef=None,is_channel=False,
             batch_size=8,catch_trial_perc=0,condition='test',go_cue_random=None,
             add_vis_noise=False,add_prop_noise=False,var_vis_noise=0.1,var_prop_noise=0.1,
             t_vis_noise=[0.1,0.15],t_prop_noise=[0.1,0.15],return_loss=False,
             disturb_hidden=False,t_disturb_hidden=0.15,d_hidden=None,batch=None):
    
    data=[]
    Loss=[]
    count = -1
    for (p,ff) in phase.items():
        for f in ff:
            count += 1

            weight_file, cfg_file, _ = get_dir(folder_name,model_name,p,f,batch=batch[count] if batch else None)
            env, policy, _, _ = load_stuff(cfg_file,weight_file,phase=p)
            ff_coefficient = f if ff_coef is None else ff_coef[count]

            data0, loss, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coefficient,is_channel=is_channel,
                                                 batch_size=batch_size,catch_trial_perc=catch_trial_perc,condition=condition,go_cue_random=go_cue_random,
                                                 add_vis_noise=add_vis_noise, add_prop_noise=add_prop_noise,
                                                 var_vis_noise=var_vis_noise, var_prop_noise=var_prop_noise,
                                                 t_vis_noise=t_vis_noise, t_prop_noise=t_prop_noise,
                                                 disturb_hidden=disturb_hidden, t_disturb_hidden=t_disturb_hidden,
                                                 d_hidden=d_hidden)
            
            loss['lateral'] = lat_dev
            loss['angle'] = ang_dev

            data.append(data0)
            Loss.append(loss)
    return (data, Loss) if return_loss else data


def get_hidden(folder_name,model_name,phase={'NF1':0},ff_coef=None,is_channel=False,demean=False,batch=None):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel,batch=batch)
    Data= []
    for i in range(len(data)):
        X = np.array(data[i]['all_hidden'])

        dims = X.shape

        if demean:
            if i==0:
                mean_dat = np.mean(X,axis=0,keepdims=True)

            X = X-mean_dat
            #X = X-np.mean(X,axis=0,keepdims=True) # TODO
            #X = X.reshape(-1,dims[-1]) # [(cond X time), neuron]
            #X = X-np.mean(X,axis=0)
            #X = np.reshape(X,newshape=dims)
        Data.append(X)
    return Data
def get_force(folder_name,model_name,phase={'NF1':0},ff_coef=None,is_channel=False,batch=None):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel,batch=batch)
    Data= []
    for i in range(len(data)):
        X = np.array(data[i]['endpoint_force'])
        Data.append(X)
    return Data

def get_vel(folder_name,model_name,phase={'NF1':0},ff_coef=None,is_channel=False):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel)
    Data= []
    for i in range(len(data)):
        X = np.array(data[i]['vel'])
        Data.append(X)
    return Data

def get_Gweights(folder_name,model_name,phase={'NF1':[0]}):
    weights = ['gru.weight_ih_l0','gru.bias_ih_l0','gru.weight_hh_l0','gru.bias_hh_l0','fc.weight','fc.bias','h0']
    weights_dict = {weight:[] for weight in weights}

    count = 0
    for i,p in enumerate(phase.keys()):
        for f in phase[p]:
            count += 1
            weight_file, _, _ = get_dir(folder_name,model_name,p,f)
            w = th.load(weight_file)
            for weight in weights:
                weights_dict[weight].append(np.ravel(w[weight].numpy()))
    
    n_cond = count
    cond_vec = np.arange(0,n_cond)
    part_vec = np.zeros_like(cond_vec)

    G = np.zeros((len(weights),n_cond,n_cond)) # Allocate memory
    rdm = np.zeros_like(G) 
    for i,weight in enumerate(weights):

        Y = np.array(weights_dict[weight])
        N , n_channel = Y.shape
        X = pcm.matrix.indicator(part_vec)
        Y -= X @ pinv(X) @ Y # Remove mean
        Z = cond_vec
        Z = indicator(Z)
        A = pinv(Z) @ Y

        G[i,:,:] = A @ A.T
        rdm[i,:,:] = pcm.G_to_dist(G[i,:,:])
    return G, rdm