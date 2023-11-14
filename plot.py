import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
from pathlib import Path
import json
from utils import *


def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def plot_training_log(log,loss_type,w=50,figsize=(10,3)):
    """
        loss_type: 'position_loss' or 'hidden_loss' or 'muscle_loss' or 'overall_loss'
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(log,dict):
       log = log[loss_type]
    log = window_average(np.array(log),w=w)

    ax.semilogy(log)

    ax.set_ylabel("Loss")
    ax.set_xlabel("Batch #")
    return fig, ax


def plot_simulations(xy, target_xy, plot_lat=True, vel=None,figsize=(8,6)):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)

    ax.scatter(target_x, target_y)

    # plot lateral deviation line
    if plot_lat:
        _, init, endp, _ = calculate_lateral_deviation(xy, target_xy)
        for i in range(8):
            ax.plot([init[i, 0], endp[i, 0]], [init[i, 1], endp[i, 1]], color='b', alpha=1, linewidth=0.5,linestyle='-')


    if vel is not None:
        # plot the line the connect initial and final positions
        for i in range(8):
            ax.plot([xy[i, 0, 0], target_xy[i, -1, 0]], [xy[i, 0, 1], target_xy[i, -1, 1]], color='k', alpha=0.2, linewidth=0.5,linestyle='--')

        # plot the line that connect initial and peak velocity positions
        vel_norm = np.linalg.norm(vel, axis=-1)
        idx = np.argmax(vel_norm, axis=1)
        xy_peakvel = xy[np.arange(xy.shape[0]), idx, :]

        for i in range(8):
            ax.plot([xy[i, 0, 0], xy_peakvel[i, 0]], [xy[i, 0, 1], xy_peakvel[i, 1]], color='k', alpha=1, linewidth=1.5,linestyle='-')
    
    return fig, ax

def plot_learning(data_dir,num_model=16,w=1000,figsize=(6,10),init_phase=1,loss_type='position_loss'):
    position_loss_NF1 = []
    position_loss_FF1 = []
    position_loss_NF2 = []
    position_loss_FF2 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)


        log_NF1 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase}_*_log.json'))[0]
        log_FF1 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+1}_*_log.json'))[0]
        log_NF2 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+2}_*_log.json'))[0]
        log_FF2 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+3}_*_log.json'))[0]
        
        log_NF1 = json.load(open(log_NF1,'r'))
        log_FF1 = json.load(open(log_FF1,'r'))
        log_NF2 = json.load(open(log_NF2,'r'))
        log_FF2 = json.load(open(log_FF2,'r'))
        
        # Append data for each model
        position_loss_NF1.append(log_NF1[loss_type])
        position_loss_FF1.append(log_FF1[loss_type])
        position_loss_NF2.append(log_NF2[loss_type])
        position_loss_FF2.append(log_FF2[loss_type])


    # Calculate window averages for all models
    if w<=1:
        NF1w = position_loss_NF1
        FF1w = position_loss_FF1
        NF2w = position_loss_NF2
        FF2w = position_loss_FF2
    else:
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
    ax[0].plot(x1w,NF1_mean,'k-',label='NF1')
    ax[0].fill_between(x1w, NF1_mean - NF1_std, NF1_mean + NF1_std, color='gray', alpha=0.5)
    ax[0].plot(x2w,FF1_mean,'g-',label='FF1')
    ax[0].fill_between(x2w, FF1_mean - FF1_std, FF1_mean + FF1_std, color='green', alpha=0.5)
    ax[0].plot(x3w,NF2_mean,'k-',label='NF2')
    ax[0].fill_between(x3w, NF2_mean - NF2_std, NF2_mean + NF2_std, color='gray', alpha=0.5)
    ax[0].plot(x4w,FF2_mean,'r-',label='FF2')
    ax[0].fill_between(x4w, FF2_mean - FF2_std, FF2_mean + FF2_std, color='red', alpha=0.5)
    ax[0].legend()


    ax[1].plot(FF1_mean,'g-',label='FF1')
    ax[1].plot(FF2_mean,'r-',label='FF2')
    ax[1].legend()


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


def plot_traj(X_latent_list, plot_scatter=1, marker=['x','o'],alpha=[1,0.5], which_times=[24], dim=3, figsize=(9, 6)):
    angle_set = np.deg2rad(np.arange(0, 360, 45))  # 8 directions
    color_list = [plt.cm.brg(cond / (2 * np.pi)) for cond in angle_set]

    if dim == 2:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i,X_latent in enumerate(X_latent_list):
            if plot_scatter:
                for tr in range(8):
                    ax.scatter(X_latent[tr, which_times, 0], X_latent[tr, which_times, 1], color=color_list[tr],marker=marker[i])
            else:
                for tr in range(8):
                    ax.plot(X_latent[tr, which_times, 0].T, X_latent[tr, which_times, 1].T, color=color_list[tr],alpha=alpha[i])

    elif dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for i,X_latent in enumerate(X_latent_list):
            if plot_scatter:
                for tr in range(8):
                    ax.scatter3D(X_latent[tr, which_times, 0], X_latent[tr, which_times, 1], X_latent[tr, which_times, 2], color=color_list[tr],marker=marker[i])
            else:
                for tr in range(8):
                    ax.plot3D(X_latent[tr, which_times, 0].T, X_latent[tr, which_times, 1].T, X_latent[tr, which_times, 2].T, color=color_list[tr],alpha=alpha[i])

    else:
        print("Dimension must be 2 or 3!")

    return fig, ax

def plot_force(endpoint1,endpoint2):
    fg, ax = plt.subplots(nrows=8,ncols=1,figsize=(10,10))

    x = np.linspace(0, 1, 100)

    for i in range(8):
        
        ax[i].plot(x,np.linalg.norm(endpoint1[i,:,:],axis=1),color='red',label='noFF')
        ax[i].plot(x,np.linalg.norm(endpoint2[i,:,:],axis=1),color='blue',label='noFF')
        
        ax[i].set_ylabel('Force [N]')
        ax[i].set_xlabel('Time [s]')
    return fg, ax
