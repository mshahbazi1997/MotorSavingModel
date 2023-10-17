import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
from pathlib import Path
import json


def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def plot_training_log(log,w=50,figsize=(10,3)):
    fig, ax = plt.subplots(figsize=figsize)
    log = window_average(np.array(log),w=w)

    ax.semilogy(log)

    ax.set_ylabel("Loss")
    ax.set_xlabel("Batch #")
    return fig, ax


def plot_simulations(xy, target_xy,figsize=(5,3)):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)

    ax.scatter(target_x, target_y)
    return fig, ax

def plot_learning(data_dir,num_model=16,w=1000,figsize=(6,10),init_phase=1):
    position_loss_NF1 = []
    position_loss_FF1 = []
    position_loss_NF2 = []
    position_loss_FF2 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)


        log_file1 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase}_*_log.json'))[0]
        log_file2 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+1}_*_log.json'))[0]
        log_file3 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+2}_*_log.json'))[0]
        log_file4 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+3}_*_log.json'))[0]
        
        position_loss_NF1_ = json.load(open(log_file1,'r'))
        position_loss_FF1_ = json.load(open(log_file2,'r'))
        position_loss_NF2_ = json.load(open(log_file3,'r'))
        position_loss_FF3_ = json.load(open(log_file4,'r'))
        
        # Append data for each model
        position_loss_NF1.append(position_loss_NF1_)
        position_loss_FF1.append(position_loss_FF1_)
        position_loss_NF2.append(position_loss_NF2_)
        position_loss_FF2.append(position_loss_FF3_)


    # Calculate window averages for all models
    NF1w = [window_average(np.array(loss), w) for loss in position_loss_NF1]
    FF1w = [window_average(np.array(loss), w) for loss in position_loss_FF1]
    NF2w = [window_average(np.array(loss), w) for loss in position_loss_NF2]
    FF2w = [window_average(np.array(loss), w) for loss in position_loss_FF2]


    # Calculate the mean and standard deviation across models
    NF1_mean = np.mean(NF1w, axis=0)
    FF1_mean = np.mean(FF1w, axis=0)
    NF2_mean = np.mean(NF2w, axis=0)
    FF2_mean = np.mean(FF2w, axis=0)

    NF1_std = np.std(NF1w, axis=0)
    FF1_std = np.std(FF1w, axis=0)
    NF2_std = np.std(NF2w, axis=0)
    FF2_std = np.std(FF2w, axis=0)

    x1w = np.arange(1,np.shape(NF1w)[1]+1)
    x2w = np.arange(1,np.shape(FF1w)[1]+1) + x1w[-1]
    x3w = np.arange(1,np.shape(NF2w)[1]+1) + x2w[-1]
    x4w = np.arange(1,np.shape(FF2w)[1]+1) + x3w[-1]


    fig,ax = plt.subplots(2,1,figsize=figsize)
    ax[0].plot(x1w,NF1_mean,'k.-',label='NF1')
    ax[0].fill_between(x1w, NF1_mean - NF1_std, NF1_mean + NF1_std, color='gray', alpha=0.5)
    ax[0].plot(x2w,FF1_mean,'g.-',label='FF1')
    ax[0].fill_between(x2w, FF1_mean - FF1_std, FF1_mean + FF1_std, color='green', alpha=0.5)
    ax[0].plot(x3w,NF2_mean,'k.-',label='NF2')
    ax[0].fill_between(x3w, NF2_mean - NF2_std, NF2_mean + NF2_std, color='gray', alpha=0.5)
    ax[0].plot(x4w,FF2_mean,'r.-',label='FF2')
    ax[0].fill_between(x4w, FF2_mean - FF2_std, FF2_mean + FF2_std, color='red', alpha=0.5)
    ax[0].legend()


    ax[1].plot(FF1_mean,'g.-',label='FF1')
    ax[1].plot(FF2_mean,'r.-',label='FF2')
    ax[1].legend()


    return fig, ax

def plot_prelearning(data_dir,num_model=16,phase=0,w=1000,figsize=(6,10)):
    position_loss_NF1 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)
        log_file1 = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_log.json'))[0]

        position_loss_NF1_ = json.load(open(log_file1,'r'))
        
        # Append data for each model
        position_loss_NF1.append(position_loss_NF1_['position_loss'])

    # Calculate window averages for all models
    NF1w = [window_average(np.array(loss), w) for loss in position_loss_NF1]

    # Calculate the mean and standard deviation across models
    NF1_mean = np.mean(NF1w, axis=0)
    NF1_std = np.std(NF1w, axis=0)


    x1w = np.arange(1,np.shape(NF1w)[1]+1)


    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(x1w,NF1_mean,'k.-',label='NF1')
    ax.fill_between(x1w, NF1_mean - NF1_std, NF1_mean + NF1_std, color='gray', alpha=0.5)
    ax.legend()

    return fig, ax 

def plot_activation(all_hidden, all_muscles):
    fg, ax = plt.subplots(nrows=8,ncols=2,figsize=(10,20))

    x = np.linspace(0, 1, 100)

    for i in range(8):
        ax[i,0].plot(x,np.array(all_muscles[i,:,:]))
        ax[i,1].plot(x,np.array(all_hidden[i,:,:]))
        
        ax[i,0].set_ylabel('muscle act (au)')
        ax[i,1].set_ylabel('hidden act (au)')
        ax[i,0].set_xlabel('time (s)')
        ax[i,1].set_xlabel('time (s)')
    return fg, ax
