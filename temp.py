# 0. Load weights
weight_file,_,_= get_dir(folder_name,model_name,'NF1',0)
W = th.load(weight_file)['fc.weight'].numpy()
U, S, Vh = np.linalg.svd(W, full_matrices=True)
V = Vh.T
P = V[:,:n_muscle] # output potent
N = V[:,n_muscle:] # output null

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






# FOR TEST!
# plot the line the connect initial and final positions
# for i in range(n_reach):
#     ax.plot([xy[i, 0, 0], target_xy[i, -1, 0]], [xy[i, 0, 1], target_xy[i, -1, 1]], color='k', alpha=0.2, linewidth=0.5,linestyle='--')

# FOR TEST!
# plot lateral deviation line
# if plot_lat:
#     _, init, endp, _ = calculate_lateral_deviation(xy, target_xy)
#     for i in range(n_reach):
#         ax.plot([init[i, 0], endp[i, 0]], [init[i, 1], endp[i, 1]], color='b', alpha=1, linewidth=0.5,linestyle='-')

# FOR TEST!
# if vel is not None:
#     # plot the line that connect initial and peak velocity positions
#     vel_norm = np.linalg.norm(vel, axis=-1)
#     idx = np.argmax(vel_norm, axis=1)
#     xy_peakvel = xy[np.arange(xy.shape[0]), idx, :]

#     for i in range(n_reach):
#         ax.plot([xy[i, 0, 0], xy_peakvel[i, 0]], [xy[i, 0, 1], xy_peakvel[i, 1]], color='k', alpha=1, linewidth=1.5,linestyle='-')



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

