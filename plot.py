import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
from pathlib import Path
import json
from utils import *




def plot_training_log(log,loss_type,w=50,figsize=(10,3)):
    """
        loss_type: 'position_loss' or 'hidden_loss' or 'muscle_loss' or 'overall_loss'
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(log,dict):
       log = log[loss_type]
    loss = window_average(np.array(log),w=w)

    ax.semilogy(loss)

    ax.set_ylabel("Loss")
    ax.set_xlabel("Batch #")
    print(np.mean(loss[:-1000]))
    return fig, ax


def plot_simulations(ax, xy, target_xy, plot_lat=True, vel=None):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    n_reach = target_xy.shape[0]

    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)

    angle_set = np.deg2rad(np.arange(0, 360, 45))  # 8 directions
    #angle_set = np.deg2rad(np.array([0,45,60,75,90,105,120,135,180,225,315]))
    color_list = [plt.cm.brg(cond / (2 * np.pi)) for cond in angle_set]

    for i in range(n_reach):
        ax.scatter(target_x[i], target_y[i],color=color_list[i])

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
    

def plot_learning(data_dir,num_model=16,phases=['NF1','FF1','NF2','FF2'],w=1,figsize=(6,10),loss_type='position',ignore=[],show_saving=False):

    color_list = ['k','g','k','r']
    if show_saving:
        fig,ax = plt.subplots(2,1,figsize=figsize)
    else: 
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax=[ax]

    loss = {phase: [] for phase in phases}
    for i,phase in enumerate(phases):
        for m in range(num_model):
            if m in ignore:
                continue
            model_name = "model{:02d}".format(m)
            log = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_log.json'))[0]
            log = json.load(open(log,'r'))
            loss[phase].append(log[loss_type])
        

        # Calculate window averages for all models
        loss[phase] = [window_average(np.array(l), w) for l in loss[phase]]
        # Calculate the mean and standard deviation across models
        loss[phase+'_mean'] = np.mean(loss[phase], axis=0)
        loss[phase+'_std'] = np.std(loss[phase], axis=0)

        loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1)
        if i > 0:
            loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1) + np.max(loss[phases[i-1]+'_x'])
            #np.shape(loss[phases[i-1]])[1]
        
        ax[0].plot(loss[phase+'_x'],loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)
        ax[0].fill_between(loss[phase+'_x'], loss[phase+'_mean'] - loss[phase+'_std'], loss[phase+'_mean'] + loss[phase+'_std'], color=color_list[i], alpha=0.5)

        if show_saving:
            if phase=='FF1' or phase=='FF2':
                ax[1].plot(loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)

    #ax[1].legend()
    ax[0].legend()
    ax[0].set_ylabel(loss_type)
    #ax[1].set_ylabel(loss_type)

    #ax[0].axhline(y=np.mean(loss['NF1_mean'][:-10]), color='k', linestyle='--', linewidth=1)

    return fig, ax


def plot_activation(all_hidden, all_muscles,figsize=(10,15)):
    fg, ax = plt.subplots(nrows=8,ncols=2,figsize=figsize)

    x = np.linspace(0, 1, np.shape(all_hidden)[1])

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

def plot_force(data,label,figsize=(10,15)):
    fg, ax = plt.subplots(nrows=8,ncols=1,figsize=figsize)

    color_list = ['m','c','g','b','r','y','k','orange']

    x = np.linspace(0, 1, np.shape(data[0]['all_endpoint'])[1])

    max_force = 0
    max_vel = 0
    for i in range(8):
        for j in range(len(data)):
            ep = np.linalg.norm(data[j]['all_endpoint'][i,:,:],axis=1)
            vel = np.linalg.norm(data[j]['vel'][i,:,:],axis=1)
            if np.max(ep)>max_force:
                max_force = np.max(ep)
            if np.max(vel)>max_vel:
                max_vel = np.max(vel)
            #ax[i,0].plot(x,ep,color=color_list[j],label=label[j])
            #ax[i,1].plot(x,vel,color=color_list[j],label='vel ' + label[j],alpha=1,linestyle='-')
            ax[i].plot(x,ep,color=color_list[j],label=label[j])
            ax[i].plot(x,8*vel,color=color_list[j],label='8 * vel ' + label[j],alpha=0.5,linestyle='--')

        #ax[i,0].axhline(y=00, color='k')
        #ax[i,1].axhline(y=00, color='k')
        ax[i].axhline(y=00, color='k')
        

        #ax[i,0].set_ylabel('Force [N]')
        #ax[i,1].set_ylabel('Velocity [m/s]')
        ax[i].set_ylabel('Force [N]')
    for i in range(8):
        #ax[i,0].set_ylim([-0.5,max_force+0.5])
        #ax[i,1].set_ylim([-0.5,max_vel+0.5])
        ax[i].set_ylim([-0.5,max_force+0.5])
    #ax[i,0].set_xlabel('Time [s]')
    #ax[i,1].set_xlabel('Time [s]')
    ax[i].set_xlabel('Time [s]')
    #ax[0,0].legend()
    #ax[0,1].legend()
    ax[0].legend()
    return fg, ax

def plot_kinematic(vel,xy,tg,figsize=(10,15)):
    """
    """
    fg, ax = plt.subplots(nrows=8,ncols=2,figsize=figsize)


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

    x = np.linspace(0, 1, np.shape(vel)[1])

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
        
    return fg, ax



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
    
def plot_epforce(data,label,figsize=(10,15)):
    fg, ax = plt.subplots(nrows=8,ncols=1,figsize=figsize)

    color_list = ['blue','orange','red','green']
    style = ['-','--','-.',':']

    x = np.linspace(0, 1, np.shape(data[0]['endpoint_force'])[1])

    max_force = 0
    min_force = 0
    for i in range(8):
        for j in range(len(data)):
            ep = np.array(data[j]['endpoint_force'][i,:,:])
            vel = np.array(data[j]['vel'][i,:,:])
    
            vel_norm = np.linalg.norm(vel,axis=-1)
            max_vel_idx = np.argmax(vel_norm)/100


            if np.max(ep)>max_force:
                max_force = np.max(ep)
            if np.min(ep)<min_force:
                min_force = np.min(ep)
            
            ax[i].plot(x,ep[:,0],color=color_list[0],label=label[j],linestyle=style[j])
            ax[i].plot(x,ep[:,1],color=color_list[1],label=label[j],linestyle=style[j])

            ax[i].axvline(x=max_vel_idx, color='k',linestyle=style[j])


        ax[i].axhline(y=00, color='k')
        ax[i].set_ylabel('Force [N]')

    for i in range(8):
        ax[i].set_ylim([-0.5+min_force,max_force+0.5])
    ax[i].set_xlabel('Time [s]')
    ax[0].legend()
    return fg, ax
