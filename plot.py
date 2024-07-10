import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
import json
from copy import deepcopy
from utils import *
from matplotlib.colors import ListedColormap
import seaborn as sb
import matplotlib.patches as patches


fontsize_label = 7
fontsize_tick = 7
fontsize_legend = 7

palette_colors = {'FF1':(0,0.5,0),'FF2':(0.4,0.4,0.8),'NF1':(0,0,0),'NF2':(0,0,0)}

color_reach = [[184/255, 130/255, 23/255, 1]] 


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

def plot_simulations(ax, xy, target_xy,cmap=None,color_dot=None,s=70,set_lim=True,plot_circle=False,set_label=False):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    n_reach = target_xy.shape[0]

    if cmap is None:
        cmap = ListedColormap(color_reach,N=1)


    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy,cmap=cmap)

    fg = ax.get_figure()
    children = fg.get_children()
    colorbar_ax = children[-1]  # Assuming the colorbar is the last axes object
    colorbar_ax.remove()


    if color_dot is None:
        colormap = plt.cm.viridis
        colors = [colormap(i/7) for i in range(8)]
    else:
        colors = np.tile(color_dot,(n_reach,1))


    for i in range(n_reach):
        ax.scatter(target_x[i], target_y[i],color=colors[i],s=s)

    

    if plot_circle:
        # Circle parameters
        start_position = np.array([1.047, 1.570])
        start_position = np.array([-0.13366797,0.43435786])

        radius = 0.1
        # Plot the circle
        circle = patches.Circle(start_position, radius, edgecolor='blue', facecolor='none', lw=2)
        ax.add_patch(circle)

    if set_lim:
        ax.set_ylim([0.3, 0.65])
        ax.set_xlim([-0.3, 0.])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(labelsize=fontsize_tick)
    ax.yaxis.set_tick_params(labelsize=fontsize_tick)

    

    ax.tick_params(left = False,bottom = False) 

    if set_label:
        ax.set_xlabel('X [m]', fontsize = fontsize_label)
        ax.set_ylabel('Y [m]', fontsize = fontsize_label)
        
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    return ax



    

def plot_learning(loss,figsize=(6,10),show_saving=False,gap=2000,ylabel='Lateral deviation [mm]'):

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


    ax[0].set_xlabel('# Batches', fontsize = fontsize_label)
    ax[0].set_ylabel(ylabel, fontsize = fontsize_label)
    ax[0].legend(title = '',frameon = False, bbox_to_anchor= (1,1), fontsize=fontsize_legend)
    ax[0].legend().set_visible(False)
    ax[0].xaxis.set_tick_params(labelsize=fontsize_tick)
    ax[0].yaxis.set_tick_params(labelsize=fontsize_tick)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    return fig, ax


def my_pointplot(T,x='phase',y='value',hue='phase',figsize=(0.8,1),xlabel='',ylabel='',ax=None,linewidth=1.5,linestyle='-',color=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if color is not None:
        palette_colors = None
    else:
        palette_colors = {'FF1':(0,0.5,0),'FF2':(0.4,0.4,0.8),'NF1':(0,0,0),'NF2':(0,0,0)}

    sb.pointplot(x=x, y=y, data=T, hue=hue, ax=ax, palette=palette_colors,errorbar=ci_func,linewidth=linewidth,linestyle=linestyle,color=color)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize_tick)
    ax.yaxis.set_tick_params(labelsize=fontsize_tick)
    ax.tick_params(bottom=False)
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    return ax

def my_barplot(T,x='size',y='value',hue='phase',figsize=(2.7, 1.8),width=0.5):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sb.barplot(x=x, y=y, data=T, hue=hue, ax=ax, palette=palette_colors,width=width,errorbar=ci_func)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize_tick)
    ax.yaxis.set_tick_params(labelsize=fontsize_tick)
    ax.tick_params(bottom=False)
    ax.set_xlabel('', fontsize=fontsize_label)
    ax.legend().set_visible(False)

    return fig, ax

def my_plot(x,y,colors,figsize,labels=None,ylim=None,plot0=False,ylabel='',linestyle=None,alpha=1):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if linestyle is None:
        linestyle = ['-']*y.shape[1]

    for i in range(y.shape[1]):
        if colors is None:
            ax.plot(x,y[:,i])
        else:
            if labels is not None:
                ax.plot(x,y[:,i],color=colors[i],label=labels[i],linewidth=2,linestyle=linestyle[i],alpha=alpha)
            else:
                ax.plot(x,y[:,i],color=colors[i],linewidth=2,linestyle=linestyle[i],alpha=alpha)

    if plot0:
        ax.axhline(y=0, color='k', linestyle='--',linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left = False,bottom = False) 

    if ylim is not None:    
        ax.set_ylim(ylim)
    ax.set_xlabel('Time [s]', fontsize = fontsize_label)
    ax.set_ylabel(ylabel, fontsize = fontsize_label)
    ax.xaxis.set_tick_params(labelsize=fontsize_tick)
    ax.yaxis.set_tick_params(labelsize=fontsize_tick)

    if labels is not None:
        ax.legend(fontsize=fontsize_legend,loc='lower right',frameon = False)

    return fig, ax


def plot_tdr(data,figsize=(2.5,2.5)):
    fig,ax = plt.subplots(1,1,figsize=figsize)

    n_cond = 8
    colormap = plt.cm.viridis
    colors = [colormap(i/7) for i in range(8)]

    s1_size = 7
    s2_size = 7
    s3_size = 7
    s4_size = 7

    s1 = 'o'
    s2 = '^'
    s3 = 's'
    s4 = '*'

    alpha = 1

    for i in range(n_cond):
        
        if i == 0:
            ax.plot(data[0][i,0], data[0][i,1],s1, markersize=s1_size, color=colors[i],alpha=alpha,label='NF1',markeredgewidth=0)
            ax.plot(data[1][i,0], data[1][i,1],s2, markersize=s2_size, markerfacecolor=colors[i],alpha=alpha,label='FF1',markeredgewidth=0)
            ax.plot(data[2][i,0], data[2][i,1],s3, markersize=s3_size, color=colors[i],alpha=alpha,label='NF2',markeredgewidth=0)
            ax.plot(data[3][i,0], data[3][i,1],s4, markersize=s4_size, markerfacecolor=colors[i],alpha=alpha,label='FF2',markeredgewidth=0)
        else:
            ax.plot(data[0][i,0], data[0][i,1],s1, markersize=s1_size, color=colors[i],alpha=alpha,markeredgewidth=0)
            ax.plot(data[1][i,0], data[1][i,1],s2, markersize=s2_size, markerfacecolor=colors[i],alpha=alpha,markeredgewidth=0)
            ax.plot(data[2][i,0], data[2][i,1],s3, markersize=s3_size, color=colors[i],alpha=alpha,markeredgewidth=0)
            ax.plot(data[3][i,0], data[3][i,1],s4, markersize=s4_size, markerfacecolor=colors[i],alpha=alpha,markeredgewidth=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left = False,bottom = False) 
    ax.set_xlabel('TDR Axis 1', fontsize = fontsize_label)
    ax.set_ylabel('TDR Axis 2', fontsize = fontsize_label)

    ax.legend(fontsize=fontsize_legend,loc='lower left',frameon = False)

    return fig, ax


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