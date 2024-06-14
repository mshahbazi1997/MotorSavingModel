import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
import json
from copy import deepcopy
from utils import *
from get_utils import get_dir


def plot_learning_curve(ax,loss,loss_type=None,w=None):
    """
        loss_type: 'position_loss' or 'hidden_loss' or 'muscle_loss' or 'overall_loss'
    """
    if isinstance(loss,dict):
       loss = loss[loss_type]
    
    if all(isinstance(item, list) for item in loss):
        loss = list(np.array(loss).mean(axis=1))

    if w is not None:
        loss = window_average(np.array(loss),w=w)

    ax.semilogy(loss)

    ax.set_ylabel('')
    ax.set_xlabel('')
    return ax


def plot_simulations(ax, xy, target_xy, plot_lat=True, vel=None,cmap='viridis',s=70):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    n_reach = target_xy.shape[0]

    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy,cmap=cmap)

    angle_set = np.deg2rad(np.arange(0, 360, 45))  # 8 directions
    #angle_set = np.deg2rad(np.array([0,45,60,75,90,105,120,135,180,225,315]))
    #color_list = [plt.cm.brg(cond / (2 * np.pi)) for cond in angle_set]
    color = np.array([0.5,0.5,0.5]) 

    for i in range(n_reach):
        ax.scatter(target_x[i], target_y[i],color=color,s=s) # color_list[i]

    # plot the line the connect initial and final positions
    for i in range(n_reach):
        ax.plot([xy[i, 0, 0], target_xy[i, -1, 0]], [xy[i, 0, 1], target_xy[i, -1, 1]], color='k', alpha=0.2, linewidth=0.5,linestyle='--')

    # plot lateral deviation line
    if plot_lat:
        _, init, endp, _ = calculate_lateral_deviation(xy, target_xy)
        for i in range(n_reach):
            ax.plot([init[i, 0], endp[i, 0]], [init[i, 1], endp[i, 1]], color='b', alpha=1, linewidth=0.5,linestyle='-')


    if vel is not None:

        # plot the line that connect initial and peak velocity positions
        vel_norm = np.linalg.norm(vel, axis=-1)
        idx = np.argmax(vel_norm, axis=1)
        xy_peakvel = xy[np.arange(xy.shape[0]), idx, :]

        for i in range(n_reach):
            ax.plot([xy[i, 0, 0], xy_peakvel[i, 0]], [xy[i, 0, 1], xy_peakvel[i, 1]], color='k', alpha=1, linewidth=1.5,linestyle='-')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left = False,bottom = False) 
    ax.set_xlabel('')
    ax.set_ylabel('')

    

def plot_learning(loss,figsize=(6,10),show_saving=False,gap=2000,palette_colors = None):

    if palette_colors is None:
        palette_colors = {'FF1': 'g', 'FF2': 'r', 'NF1': 'k', 'NF2': 'k'}

    if show_saving:
        fig,ax = plt.subplots(2,1,figsize=figsize)
    else:
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax=[ax]

    xticks = []
    xticklabels = []
    current_x = 0

    loss2 = deepcopy(loss)

    phases = list(loss.keys())
    for i,phase in enumerate(phases):

        # Calculate the mean and standard deviation across models
        loss2[phase+'_mean'] = np.mean(loss2[phase], axis=0)
        loss2[phase+'_std'] = np.std(loss2[phase], axis=0)

        # For x-ticks, show the start and end of each batch range
        batch_size = np.shape(loss2[phase])[1]
        xticks.extend([current_x, current_x + batch_size])
        xticklabels.extend(['0', str(batch_size-1)])

        x = np.arange(0, batch_size)

        ax[0].plot(x + current_x,loss2[phase+'_mean'],color=palette_colors[phase],linestyle='-',label=phase,linewidth=3)
        ax[0].fill_between(x + current_x, loss2[phase+'_mean'] - loss2[phase+'_std'], loss2[phase+'_mean'] + loss2[phase+'_std'], color=palette_colors[phase], alpha=0.5)

        current_x += batch_size + gap

        if show_saving:
            if phase=='FF1' or phase=='FF2':
                ax[1].plot(loss2[phase+'_mean'],color=palette_colors[phase],linestyle='-',label=phase)
    ax[0].set_xticks(ticks=xticks, labels=xticklabels)
    ax[0].legend()
    ax[0].axhline(y=np.mean(loss2['NF1_mean'][-10:]), color='k', linestyle='--', linewidth=1)

    return fig, ax


def plot_force(ax,force_mag,vel_mag,color='b',dt=0.01,b=8,lw=4):
    """
        force_mag: (n_reach, n_time)
        vel_mag: (n_reach, n_time)
    """
    
    n_reach = np.shape(force_mag)[0]
    x = np.linspace(0, np.shape(force_mag)[1]*dt, np.shape(force_mag)[1])

    max_force = 0
    max_vel = 0

    for i in range(n_reach):

        ep = force_mag[i,:]
        vel = vel_mag[i,:]

        if np.max(ep)>max_force:
            max_force = np.max(ep)

        if np.max(vel)>max_vel:
            max_vel = np.max(vel)

        ax[i].plot(x,ep,label='Force',color=color,linewidth=lw)
        ax[i].plot(x,b*vel,label='Ideal',alpha=0.5,linestyle='--',color=color,linewidth=lw)

        ax[i].axhline(y=00,color='k')
        ax[i].set_ylabel('Force [N]')

    for i in range(n_reach):
        ax[i].set_ylim([-0.5,max_vel*b+0.5])
    ax[i].set_xlabel('Time [s]')
    ax[0].legend()
    return ax

def plot_kinematic(ax,vel,xy,tg,dt=0.01):
    """
    """


    vel = np.array(vel)
    xy = np.array(xy)
    tg = np.array(tg)

    # use for calculating lateral and parallel speed
    #target = tg[:,-1,:]
    #init = xy[:,0,:]

    #line_vector = target - init
    #line_vector2 = np.tile(line_vector[:,None,:],(1,vel.shape[1],1))

    #projection = np.sum(line_vector2 * vel, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
    #projection = line_vector2 * projection[:,:,np.newaxis]

    #lat_speed = vel - projection

    #vel = lat_speed
    #vel = projection

    color_list = ['blue','orange','red','green']

    x = np.linspace(0, np.shape(vel)[1]*dt, np.shape(vel)[1])

    for i in range(8):

        ax[i,0].plot(x,xy[i,:,0],color=color_list[2],label='x')
        ax[i,0].plot(x,xy[i,:,1],color=color_list[3],label='y')
        ax[i,0].plot(x,tg[i,:,0],color=color_list[0])
        ax[i,0].plot(x,tg[i,:,1],color=color_list[1])
        ax[i,0].set_ylabel('position [m]')

        ax[i,1].plot(x,vel[i,:,0],color=color_list[0],label='x')
        ax[i,1].plot(x,vel[i,:,1],color=color_list[1],label='y')
        ax[i,1].axhline(y=0, color='k')
        ax[i,1].set_ylabel('velocity [m/s]')
        

    ax[i,0].legend()
    ax[i,1].legend()
    
    ax[i,0].set_xlabel('Time [s]')
    ax[i,1].set_xlabel('Time [s]')
        
    return ax


def plot_traj(ax,X_latent_list, plot_scatter=1, marker=['x', 'o', '*', 's', 'D', 'v', '>', '<', 'p', '^'],alpha=[1,0.5,0.4,0.2], which_times=[24], dim=3, which_latent=[0,1,2]):
    angle_set = np.deg2rad(np.arange(0, 360, 45))  # 8 directions
    color_list = [plt.cm.brg(cond / (2 * np.pi)) for cond in angle_set]

    if dim == 2:
        for i,X_latent in enumerate(X_latent_list):
            if plot_scatter:
                for tr in range(8):
                    ax.scatter(X_latent[tr, which_times, which_latent[0]], X_latent[tr, which_times, which_latent[1]], color=color_list[tr],marker=marker[i])
            else:
                for tr in range(8):
                    ax.plot(X_latent[tr, which_times, which_latent[0]].T, X_latent[tr, which_times, which_latent[1]].T, color=color_list[tr],alpha=alpha[i])

    elif dim == 3:
        for i,X_latent in enumerate(X_latent_list):
            if plot_scatter:
                for tr in range(8):
                    ax.scatter3D(X_latent[tr, which_times, which_latent[0]], X_latent[tr, which_times, which_latent[1]], X_latent[tr, which_times, which_latent[2]], color=color_list[tr],marker=marker[i])
            else:
                for tr in range(8):
                    ax.plot3D(X_latent[tr, which_times, which_latent[0]].T, X_latent[tr, which_times, which_latent[1]].T, X_latent[tr, which_times, which_latent[2]].T, color=color_list[tr],alpha=alpha[i])

    else:
        print("Dimension must be 2 or 3!")

    #return fig, ax


def plot_Gs(G,grid = None,labels=[],titles=[],figsize=(12,12),vmin=None, vmax=None):
    numG,n_cond,n_cond = G.shape
    if grid is None:
        a = int(np.ceil(np.sqrt(numG)))
        b = int(np.ceil(numG/a))
        grid = (a,b)
    
    plt.figure(figsize=figsize)
    for i in range(numG):
        
        plt.subplot(grid[0],grid[1],i+1)
        plt.title(titles[i])
        plt.imshow(G[i],vmin=vmin,vmax=np.max(G[i]))
        plt.colorbar()

        plt.xticks(np.arange(n_cond),labels)
        plt.yticks(np.arange(n_cond),labels)