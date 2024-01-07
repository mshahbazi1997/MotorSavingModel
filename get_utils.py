import os
from pathlib import Path
from utils import load_stuff
from model import test
import numpy as np

base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def get_dir(folder_name,model_name,phase):

    data_dir = os.path.join(base_dir,folder_name)

    weight_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_weights'))[0]
    cfg_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_cfg.json'))[0]
    loss_file = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_log.json'))[0]

    return weight_file, cfg_file, loss_file

def get_data(folder_name,model_name,phase=['NF1'],ff_coef=[0],is_channel=False):
    import warnings
    warnings.filterwarnings('ignore')
    data=[]
    for i,p in enumerate(phase):
        weight_file, cfg_file, _ = get_dir(folder_name,model_name,p)
        
        env, policy, _, _ = load_stuff(cfg_file,weight_file,phase=phase)
        data0, loss, ang_dev, lat_dev = test(env,policy,ff_coefficient=ff_coef[i],is_channel=is_channel)

        data.append(data0)
    return data

def get_hidden(folder_name,model_name,phase=['NF1'],ff_coef=[0],is_channel=False,demean=False):
    data = get_data(folder_name,model_name,phase,ff_coef,is_channel)
    Data= []
    for i in range(len(data)):
        X = np.array(data[i]['all_hidden'])

        dims = X.shape

        if demean:
            X = X-np.mean(X,axis=0,keepdims=True) # TODO
            X = X.reshape(-1,dims[-1]) # [(cond X time), neuron]
            X = X-np.mean(X,axis=0)
            X = np.reshape(X,newshape=dims)
        Data.append(X)
    return Data
def get_weights():
    1==1