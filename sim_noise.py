import os
import sys 
sys.path.append(os.path.abspath('..'))
import json
from pathlib import Path
from get_utils import get_data,get_dir
from model import test
import plot as plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed




def sim_prop(i):
    folder_name = 'Sim_all_inv2'
    model_name = 'model01'

    phase = {'NF1':[0],'NF2':[0]}
    ff_coef = None#[8,8]
    var=[0,0.1]

    T = pd.DataFrame()
    for v in var:
        for t in np.linspace(0, 0.95, 20):
            if var==0:
                add_prop_noise=False
            else:
                add_prop_noise=True
            data, loss = get_data(folder_name,model_name,phase,ff_coef,is_channel=False,add_prop_noise=add_prop_noise,var_prop_noise=v,t_prop_noise=[t,t+0.02],return_loss=True)

            for j,key in enumerate(phase.keys()):
                l = loss[j]['position'].item()
                d = {'t':[t],'loss':[l],'phase':[key],'noise':[v]}
                T = pd.concat([T,pd.DataFrame(d)],ignore_index=True)
    T.to_csv(f'simulations/sim{i}_prop_noise.csv',index=False)

def sim_vis(i):
    folder_name = 'Sim_all_inv2'
    model_name = 'model01'

    phase = {'NF1':[0],'NF2':[0]}
    ff_coef = None#[8,8]
    var=[0,0.5]

    T = pd.DataFrame()
    for v in var:
        for t in np.linspace(0, 0.95, 20):
            if var==0:
                add_vis_noise=False
            else:
                add_vis_noise=True
            data, loss = get_data(folder_name,model_name,phase,ff_coef,is_channel=False,add_vis_noise=add_vis_noise,var_vis_noise=v,t_vis_noise=[t,t+0.02],return_loss=True)

            for j,key in enumerate(phase.keys()):
                l = loss[j]['position'].item()
                d = {'t':[t],'loss':[l],'phase':[key],'noise':[v]}
                T = pd.concat([T,pd.DataFrame(d)],ignore_index=True)
    T.to_csv(f'simulations/sim{i}_vis_noise.csv',index=False)

if __name__ == "__main__":
    noise_type = 'vis'
    iter_list = range(50) 
    num_processes = len(iter_list)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if noise_type=='prop':
            futures = {executor.submit(sim_prop, iteration): iteration for iteration in iter_list}
        elif noise_type=='vis':
            futures = {executor.submit(sim_vis, iteration): iteration for iteration in iter_list}
        for future in as_completed(futures):
          try:
            result = future.result()
          except Exception as e:
            print(f"Error in iteration {futures[future]}: {e}")
    # once all processes are done, 
    # load all .cvs files and plot
    L = pd.DataFrame()
    for i in iter_list:
       T = pd.read_csv(f'simulations/sim{i}_{noise_type}_noise.csv')
       # delete this file once loaded
       os.remove(f'simulations/sim{i}_{noise_type}_noise.csv')
       T.sim = i
       L = pd.concat([L,T],ignore_index=True)
    L.to_csv(f'simulations/sim_{noise_type}_noise.csv',index=False)


