import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
from pathlib import Path
import json
from utils import *
from get_utils import get_dir


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
    print(np.mean(loss[:-int(1000/w)]))
    return fig, ax


def plot_simulations(ax, xy, target_xy, plot_lat=True, vel=None,cmap='viridis'):
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
        ax.scatter(target_x[i], target_y[i],color=color,s=70) # color_list[i]

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
    

def plot_learning(folder_name,num_model=16,phases={'NF1':[0],'FF1':[8],'NF2':[0],'FF2':[8]},w=1,figsize=(6,10),loss_type='position',ignore=[],show_saving=False,gap=2000):

    all_phase = list(phases.keys())

    color_list = ['k','g','k','r']
    if show_saving:
        fig,ax = plt.subplots(2,1,figsize=figsize)
    else:
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax=[ax]

    loss = {phase: [] for phase in phases.keys()}

    xticks = []
    xticklabels = []
    current_x = 0
    for i,phase in enumerate(phases.keys()):
        for m in range(num_model):
            #print(m)
            if m in ignore:
                continue
            model_name = "model{:02d}".format(m)
            _,_,log=get_dir(folder_name, model_name, phase, phases[phase][0])
            log = json.load(open(log,'r'))
            loss[phase].append(log[loss_type])

        # Calculate window averages for all models
        loss[phase] = [window_average(np.array(l), w) for l in loss[phase]]
        # Calculate the mean and standard deviation across models
        loss[phase+'_mean'] = np.mean(loss[phase], axis=0)
        loss[phase+'_std'] = np.std(loss[phase], axis=0)

        # For x-ticks, show the start and end of each batch range
        batch_size = np.shape(loss[phase])[1]
        xticks.extend([current_x, current_x + batch_size])
        xticklabels.extend(['1', str(batch_size-1)])

        x = np.arange(0, batch_size)

        

        #loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1)
        #if i > 0:
            #loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1) + np.max(loss[all_phase[i-1]+'_x']) + gap
            #np.shape(loss[phases[i-1]])[1]
        
        #ax[0].plot(loss[phase+'_x'],loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)
        #ax[0].fill_between(loss[phase+'_x'], loss[phase+'_mean'] - loss[phase+'_std'], loss[phase+'_mean'] + loss[phase+'_std'], color=color_list[i], alpha=0.5)

        ax[0].plot(x + current_x,loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase,linewidth=3)
        ax[0].fill_between(x + current_x, loss[phase+'_mean'] - loss[phase+'_std'], loss[phase+'_mean'] + loss[phase+'_std'], color=color_list[i], alpha=0.5)

        current_x += batch_size + gap

        if show_saving:
            if phase=='FF1' or phase=='FF2':
                ax[1].plot(loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)
    ax[0].set_xticks(ticks=xticks, labels=xticklabels)
    #ax[1].legend()
    ax[0].legend()
    #ax[0].set_yscale('log')

    ax[0].set_ylabel(loss_type)
    #ax[1].set_ylabel(loss_type)

    ax[0].axhline(y=np.mean(loss['NF1_mean'][-10:]), color='k', linestyle='--', linewidth=1)

    return fig, ax
def plot_learning_perturbation(folder_name,num_model=20,phases={'NF1':[0],'FF1':[8],'NF2':[0],'FF2':[8]},w=1,figsize=(6,10),loss_type='position',ignore=[],show_saving=False):

    all_phase = list(phases.keys())

    color_list = ['k','g','k','r','m','c']
    if show_saving:
        fig,ax = plt.subplots(2,1,figsize=figsize)
    else: 
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax=[ax]

    #loss = {'NF1':[],'FF1':[],'NF2':[],'FF2_-1':[],'FF2_0':[],'FF2_1':[]}
    loss = {'NF1':[],'FF1':[],'NF2':[],'FF2_0':[],'FF2_-1':[],'FF2_1':[]}

    for i,phase in enumerate(['NF1','FF1','NF2','FF2_0','FF2_-1','FF2_1']):
        for m in range(num_model):
            #print(m)
            if m in ignore:
                continue
            model_name = "model{:02d}".format(m)
            
            if phase == 'FF2_-1' or phase == 'FF2_0' or phase == 'FF2_1':
                phase_temp = 'FF2'
                _,_,log=get_dir(folder_name, model_name, phase_temp, phases[phase_temp][0])
                log = json.load(open(log,'r'))
                if phase == 'FF2_-1':
                    log = log['-1']
                elif phase == 'FF2_0':
                    log = log['0']
                elif phase == 'FF2_1':
                    log = log['1']
                loss[phase].append(log[loss_type])
            else:
                _,_,log=get_dir(folder_name, model_name, phase, phases[phase][0])
                log = json.load(open(log,'r'))
                loss[phase].append(log[loss_type])

        # Calculate window averages for all models
        loss[phase] = [window_average(np.array(l), w) for l in loss[phase]]
        # Calculate the mean and standard deviation across models
        loss[phase+'_mean'] = np.mean(loss[phase], axis=0)
        loss[phase+'_std'] = np.std(loss[phase], axis=0)

        loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1)
        if i > 0:
            if i>=3:
                loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1) + np.max(loss[all_phase[2]+'_x'])
            else:
                loss[phase+'_x'] = np.arange(1,np.shape(loss[phase])[1]+1) + np.max(loss[all_phase[i-1]+'_x'])
            #np.shape(loss[phases[i-1]])[1]
        
        ax[0].plot(loss[phase+'_x'],loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)
        ax[0].fill_between(loss[phase+'_x'], loss[phase+'_mean'] - loss[phase+'_std'], loss[phase+'_mean'] + loss[phase+'_std'], color=color_list[i], alpha=0.5)

        if show_saving:
            if phase=='FF1' or phase=='FF2_0' or phase=='FF2_-1' or phase=='FF2_1':
                ax[1].plot(loss[phase+'_mean'],color=color_list[i],linestyle='-',label=phase)

    #ax[1].legend()
    ax[0].legend()
    #ax[0].set_yscale('log')

    ax[0].set_ylabel(loss_type)
    #ax[1].set_ylabel(loss_type)

    ax[0].axhline(y=np.mean(loss['NF1_mean'][-10:]), color='k', linestyle='--', linewidth=1)

    return fig, ax



def plot_activation(all_hidden, all_muscles,figsize=(10,15),dt=0.01):
    fg, ax = plt.subplots(nrows=8,ncols=2,figsize=figsize)

    x = np.linspace(0, np.shape(all_hidden)[1]*dt, np.shape(all_hidden)[1])

    for i in range(8):
        ax[i,0].plot(x,np.array(all_muscles[i,:,:]))
        ax[i,1].plot(x,np.array(all_hidden[i,:,:]))
        
        ax[i,0].set_ylabel('muscle act (au)')
        ax[i,1].set_ylabel('hidden act (au)')
    ax[i,0].set_xlabel('time (s)')
    ax[i,1].set_xlabel('time (s)')
    return fg, ax


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

def plot_force(ax,endpoint_load,velocity,color='b',dt=0.01):
    
    n_reach = np.shape(endpoint_load)[0]
    x = np.linspace(0, np.shape(endpoint_load)[1]*dt, np.shape(endpoint_load)[1])

    max_force = 0
    max_vel = 0
    for i in range(n_reach):

        #ep = np.linalg.norm(endpoint_load[i,:,:],axis=1)
        #vel = np.linalg.norm(velocity[i,:,:],axis=1)
        ep = endpoint_load[i,:]
        vel = velocity[i,:]
        if np.max(ep)>max_force:
            max_force = np.max(ep)
        if np.max(vel)>max_vel:
            max_vel = np.max(vel)

        ax[i].plot(x,ep,label='force',color=color,linewidth=5)
        ax[i].plot(x,8*vel,label='8 * vel',alpha=0.5,linestyle='--',color=color,linewidth=5)

        ax[i].axhline(y=00,color='k')
        ax[i].set_ylabel('Force [N]')

    for i in range(n_reach):
        ax[i].set_ylim([-0.5,max_vel*8+0.5])
    ax[i].set_xlabel('Time [s]')
    ax[0].legend()
    return ax

def plot_kinematic(vel,xy,tg,figsize=(10,15),dt=0.01):
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
    
def plot_epforce(data,label,figsize=(10,15),dt=0.01):
    fg, ax = plt.subplots(nrows=8,ncols=1,figsize=figsize)

    color_list = ['blue','orange','red','green']
    style = ['-','--','-.',':']

    x = np.linspace(0, np.shape(data[0]['endpoint_force'])[1]*dt, np.shape(data[0]['endpoint_force'])[1])

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
