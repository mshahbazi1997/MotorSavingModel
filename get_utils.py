import os
from pathlib import Path
from utils import load_stuff
from model import test
import numpy as np

base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def get_dir(folder_name,model_name,phase,ff_coef):
    """
    Get the directory of the model with the given parameters
        params:
            ff_coef: the coeffient by which we train the model, can be found in the phase dictionary
    """

    data_dir = os.path.join(base_dir,folder_name)
    weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_weights'))[0]
    cfg_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_cfg.json'))[0]
    loss_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_FFCoef={ff_coef}_log.json'))[0]
        
    return weight_file, cfg_file, loss_file

def get_data(folder_name,model_name,phase={'NF1':[0]},ff_coef=None,is_channel=False,
             add_vis_noise=False, add_prop_noise=False, var_vis_noise=0.1, var_prop_noise=0.1,
             t_vis_noise=[0.1,0.15], t_prop_noise=[0.1,0.15],return_loss=False):
    # here i want to add a noise option

    data=[]
    Loss=[]
    count = -1
    for i,p in enumerate(phase.keys()):
        for f in phase[p]:
            count += 1
            weight_file, cfg_file, _ = get_dir(folder_name,model_name,p,f)
            
            env, policy, _, _ = load_stuff(cfg_file,weight_file,phase=p)
            if ff_coef is None:
                data0, loss, ang_dev, lat_dev = test(env,policy,ff_coefficient=f,is_channel=is_channel,
                                                     add_vis_noise=add_vis_noise, add_prop_noise=add_prop_noise,
                                                     var_vis_noise=var_vis_noise, var_prop_noise=var_prop_noise,
                                                     t_vis_noise=t_vis_noise, t_prop_noise=t_prop_noise)
            else:
                data0, loss, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coef[count],is_channel=is_channel,
                                                     add_vis_noise=add_vis_noise, add_prop_noise=add_prop_noise,
                                                     var_vis_noise=var_vis_noise, var_prop_noise=var_prop_noise,
                                                     t_vis_noise=t_vis_noise, t_prop_noise=t_prop_noise)

            data.append(data0)
            Loss.append(loss)
    if return_loss:
        return data, Loss
    else:
        return data

def get_hidden(folder_name,model_name,phase={'NF1':0},ff_coef=None,is_channel=False,demean=False):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel)
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
def get_force(folder_name,model_name,phase={'NF1':0},ff_coef=None,is_channel=False):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel)
    Data= []
    for i in range(len(data)):
        X = np.array(data[i]['endpoint_force'])
        Data.append(X)
    return Data


def get_weights():
    1==1