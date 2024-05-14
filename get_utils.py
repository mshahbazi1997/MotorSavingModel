import os
from pathlib import Path
from utils import load_stuff
from model import test
import numpy as np
import re

#import torch as th
#import PcmPy as pcm
#from PcmPy.matrix import indicator
#from scipy.linalg import pinv


base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def get_dir(folder_name,model_name,phase,ff_coef,batch=None):
    """
    Get the directory of the model with the given parameters
        params:
            ff_coef: the coeffient by which we train the model, can be found in the phase dictionary
    """

    data_dir = os.path.join(base_dir,folder_name)

    try:
        cfg_files = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_cfg.json'))
        cfg_file = cfg_files[0] if cfg_files else None

        loss_files = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_log.json'))
        loss_file = loss_files[0] if loss_files else None

        if batch is None:
            weight_files = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_weights*'))
        else:
            weight_files = list(Path(data_dir).glob(f'{model_name}_phase={phase}_batch={batch}_FFCoef={ff_coef}_weights*'))
        weight_file = weight_files[0] if weight_files else None

        # Check if any file does not exist
        if not cfg_file or not loss_file or not weight_file:
            found = False
        else:
            found = True
    except Exception as e:
        cfg_file = None
        loss_file = None
        weight_file = None

        found = False

    # cfg_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_cfg.json'))[0]
    # loss_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_log.json'))[0]
    # if batch is None:
    #     weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_weights'))[0]
    # else:
    #     weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_batch={batch}_FFCoef={ff_coef}_weights'))[0]
    return weight_file, cfg_file, loss_file

def get_data(folder_name,model_name,phase={'NF1':[0]},ff_coef=None,is_channel=False,
             batch_size=8,catch_trial_perc=0,condition='test',go_cue_random=None,return_loss=False,
             disturb_hidden=False,t_disturb_hidden=0.15,d_hidden=None,batch=None,seed=None,n_hidden=128):
    
    data=[]
    Loss=[]
    count = -1
    for (p,ff) in phase.items():
        for f in ff:
            count += 1

            weight_file, cfg_file, _ = get_dir(folder_name,model_name,p,f,batch=batch[count] if batch else None)
            env, policy, _, _ = load_stuff(cfg_file,weight_file,phase=p,n_hidden=n_hidden)
            ff_coefficient = f if ff_coef is None else ff_coef[count]

            data0, loss, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coefficient,is_channel=is_channel,
                                                 batch_size=batch_size,catch_trial_perc=catch_trial_perc,condition=condition,go_cue_random=go_cue_random,
                                                 disturb_hidden=disturb_hidden, t_disturb_hidden=t_disturb_hidden,
                                                 d_hidden=d_hidden, seed=seed)
            
            loss['lateral'] = lat_dev
            loss['angle'] = ang_dev

            data.append(data0)
            Loss.append(loss)
    return (data, Loss) if return_loss else data

def return_ignore(folder_name,num_model,phase='FF2',ff_coef=8):
    ignore = []
    data_dir = os.path.join(base_dir,folder_name)
    
    loss_files = list(Path(data_dir).glob(f'*_phase={phase}_FFCoef={ff_coef}_log.json'))

    models = []

    for file_path in loss_files:
        file_name = file_path.name
        model_number_match = re.search(r'model(\d+)_phase',file_name)
        if model_number_match:
            models.append(int(model_number_match.group(1)))

    #print(models)
    ignore = [model for model in range(num_model) if model not in models]

    
    return ignore

def get_loss(folder_name,num_model,phases,loss_type='position',w=1,target=None,ignore=[]):
    from utils import window_average
    import json

    # load behavior and start fitting
    loss = {phase: [] for phase in phases.keys()}
    for i,phase in enumerate(phases.keys()):
        for m in range(num_model):
            if m in ignore:
                continue
            model_name = "model{:02d}".format(m)
            _,_,log=get_dir(folder_name, model_name, phase, phases[phase][0])

            log = json.load(open(log,'r'))
            
            if all(isinstance(item, list) for item in log[loss_type]):
                if target is None:
                    loss[phase].append(list(np.array(log[loss_type]).mean(axis=1)))
                else:
                    loss[phase].append(list(np.array(log[loss_type])[:,target]))
            else:    
                loss[phase].append(log[loss_type])

        # Calculate window averages for all models
        if w>1:
            loss[phase] = [window_average(np.array(l), w) for l in loss[phase]]

    return loss




# def get_weights_G(folder_name,model_name,phase={'NF1':[0]}):
#     weights = ['gru.weight_ih_l0','gru.bias_ih_l0','gru.weight_hh_l0','gru.bias_hh_l0','fc.weight','fc.bias','h0']
#     weights_dict = {weight:[] for weight in weights}

#     count = 0
#     for i,p in enumerate(phase.keys()):
#         for f in phase[p]:
#             count += 1
#             weight_file, _, _ = get_dir(folder_name,model_name,p,f)
#             w = th.load(weight_file)
#             for weight in weights:
#                 weights_dict[weight].append(np.ravel(w[weight].numpy()))
    
#     n_cond = count
#     cond_vec = np.arange(0,n_cond)
#     part_vec = np.zeros_like(cond_vec)

#     G = np.zeros((len(weights),n_cond,n_cond)) # Allocate memory
#     rdm = np.zeros_like(G) 
#     for i,weight in enumerate(weights):

#         Y = np.array(weights_dict[weight])
#         N , n_channel = Y.shape
#         X = pcm.matrix.indicator(part_vec)
#         Y -= X @ pinv(X) @ Y # Remove mean
#         Z = cond_vec
#         Z = indicator(Z)
#         A = pinv(Z) @ Y

#         G[i,:,:] = A @ A.T
#         rdm[i,:,:] = pcm.G_to_dist(G[i,:,:])
#     return G, rdm