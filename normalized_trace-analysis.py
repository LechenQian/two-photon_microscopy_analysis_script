#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:05:51 2021

@author: lechenqian
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import watershed, dilation
from skimage.segmentation import find_boundaries
from skimage.io import imsave, imread
from skimage.feature import peak_local_max
import pickle
import sys
from skimage import io
import time
import h5py
import os
import math
import pandas as pd
import re
import seaborn as sns
import statsmodels
import scipy
import pickle5
import json
from collections import Counter
#%%

def pickle_dict(df, path, filename):
    try:
        os.makedirs(path)  # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path + '/{}.pickle'.format(filename)
    with open(Path(filename), 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save to pickle done!')

def load_pickleddata(filename):
    with open(Path(filename), 'rb') as handle:
        try:
            df = pickle.load(handle)
        except:
            df = pickle5.load(handle)

    return df


def pickle_dict(df, path, filename):
    try:
        os.makedirs(path)  # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path + '/{}.pickle'.format(filename)
    with open(Path(filename), 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save to pickle done!')


def stack_traces_from_multi_mice(mice_name:list, data:dict, keyname :str, trialtype: str,stack:str,partial,index_list):
    traces_tensor_list = [] 

    for mouse in mice_name:
        if not partial:
            temp_list = [item for item in data[mouse][keyname][trialtype]]
            if stack == 'axis2':
                stacked_multisesions_traces = np.dstack(temp_list)
            elif stack == 'axis1':
                stacked_multisesions_traces = np.hstack(temp_list)
            elif stack == 'axis0':
                stacked_multisesions_traces = np.vstack(temp_list)
                
                
        else:
            
            temp_list = [data[mouse][keyname][trialtype][i] for i in index_list]
            try:
                if stack == 'axis2':
                    stacked_multisesions_traces = np.dstack(temp_list)
                elif stack == 'axis1':
                    stacked_multisesions_traces = np.hstack(temp_list)
                elif stack == 'axis0':
                    stacked_multisesions_traces = np.vstack(temp_list)
            except:
                stacked_multisesions_traces = temp_list.copy()
        traces_tensor_list.append(stacked_multisesions_traces)
    trace_tensor_multi = np.concatenate(traces_tensor_list, axis = 0 )
    return trace_tensor_multi


def stack_sliced_avgtraces_from_multi_mice(mice_name:list, data:dict, keyname :str, trialtype: str,stack:str,slice_range:list):
    traces_tensor_list = [] 

    for mouse in mice_name:
        
        temp_list = [ item[:,slice_range[0]:slice_range[1]] for item in data[mouse][keyname][trialtype]]
        if stack == 'axis2':
            stacked_multisesions_traces = np.dstack(temp_list)
        elif stack == 'axis1':
            stacked_multisesions_traces = np.hstack(temp_list)
        elif stack == 'axis0':
            stacked_multisesions_traces = np.vstack(temp_list)
                
                
        
        traces_tensor_list.append(stacked_multisesions_traces)
    trace_tensor_multi = np.concatenate(traces_tensor_list, axis = 0 )
    return trace_tensor_multi


def filter_assignment(mat,n_reg):
    assignments_filtered = np.array(mat[np.sum(~np.isnan(mat), axis=1) >= n_reg], dtype=int)
    return assignments_filtered
def getKeys(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

#%%

def plot_heatmap2(avg_mat,odor_on_frame,window_pre,window_post,line_position,is_save,title):
    
    
    plt.figure(figsize = (3,7))
    
    if window_pre == 5:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+15)],axis = 1)
    else:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+10)],axis = 1)
    order = np.argsort(-peak)
    sns.heatmap(avg_mat[order,:],vmin = -5,vmax = 5,cmap = 'RdBu_r')
    for pos in line_position:
        plt.axvline(x = int(np.ceil(pos)),color = 'grey')
    
    plt.ylabel('Neuron #')
    plt.xlabel('Time(s)')
    plt.xticks(np.arange(0, 46, step=5),[0,1,2,3,4,5,6,7,8,9])
    plt.yticks(np.arange(0, avg_mat.shape[0]+1, step=50),np.arange(0, avg_mat.shape[0]+1, step=50))
    plt.title(title)
    if is_save:
        plt.savefig('figures/{}'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
    plt.show()

def plot_heatmap_concat(avg_mat,window_pre,line_position,is_save,title,vmin=-5,vmax = 5):
    
    
    plt.figure(figsize = (12,int(avg_mat.shape[0]/30)))
    
    if window_pre == 5:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+15)],axis = 1)
    else:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+15)],axis = 1)
    order = np.argsort(-peak)
    sns.heatmap(avg_mat[order,:],vmin = vmin,vmax = vmax,cmap = 'RdBu_r')
    for pos in line_position:
        plt.axvline(x = int(np.ceil(pos)),color = 'grey')
    
    plt.ylabel('Neuron #')
    plt.xlabel('Time(s)')
    ticks = [0,1,2,3,4,5,6,7] * 6
    plt.xticks(np.arange(0, 5*8*6, step=5),ticks)
    plt.yticks(np.arange(0, avg_mat.shape[0]+1, step=50),np.arange(0, avg_mat.shape[0]+1, step=50))
    plt.title(title)
    if is_save:
        plt.savefig('figures/{}.png'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
    plt.show()
    
def plot_heatmap_concat_nonsort(avg_mat,window_pre,line_position,h_line_position,vlim, is_save,title):
    
    
    plt.figure(figsize = (12,7)) # 30

    sns.heatmap(avg_mat,vmin = -vlim,vmax = vlim,cmap = 'RdBu_r')
    for pos in line_position:
        plt.axvline(x = int(np.ceil(pos)),color = 'grey')
    for pos in h_line_position:
        plt.axhline(y = int(np.ceil(pos)),color = 'cyan')
    
    plt.ylabel('Neuron #')
    plt.xlabel('Time(s)')
    ticks = [0,1,2,3,4,5,6,7] * 6
    plt.xticks(np.arange(0, 5*8*6, step=5),ticks)
    plt.yticks(np.arange(0, avg_mat.shape[0]+1, step=50),np.arange(0, avg_mat.shape[0]+1, step=50))
    plt.title(title)
    if is_save:
        plt.savefig('figures/{}.png'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
    plt.show()
def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('cool')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

#%% load data
filename ="/Users/lechenqian/Dropbox (Uchida Lab)/Analyze_imaging_data/partial_to_mac_2021-10-23_registration_and_registerd_matrices_added.pickle.pickle"
partial_dict = load_pickleddata(filename)

#%%% define mouse group 1
# deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d1 = ['D1-05','D1-02','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']
deg_d1_notr  = ['D1-01','D1-03','D1-09']
#%%% define mouse group 2
# deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d1 = ['D1-02','D1-12','D1-13'] #without D1-05
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']
deg_d1_notr  = ['D1-01','D1-03','D1-09']
#%% stack 2 session from all mice in a group
mat_deg_d1 = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d2 = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [0,40])
mat_con_d1 = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [0,40])
mat_con_d2 = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [0,40])

#%% plot unclustered tracked neurons
mat = np.vstack([mat_deg_d1,mat_con_d1])
# mat = mat_con_d1

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat(mat,8,line_position,False ,'tracked deg+con d1')

#%% normalize by the response of each cell's maximum response across all sessions (not deal with cue yet)

#get the maximum absolute response in the odor response period
# deg d2
response_period_deg = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [10,40])

n_mat_deg_d2 = (mat_deg_d2.T/np.max(np.abs(response_period_deg[:,:90]),axis = 1)).T

# plot_heatmap_concat(n_mat_deg_d2,5,[0,0],False,'d2_normalized',vmin = -1.5,vmax = 1.5)
print(n_mat_deg_d2.shape)
# cond d2
response_period_cond = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [10,40])

n_mat_con_d2 = (mat_con_d2.T/np.max(np.abs(response_period_cond[:,:90]),axis = 1)).T
print(n_mat_con_d2.shape)



# concat the two d2 mat
concat_d2 = np.vstack([n_mat_deg_d2,n_mat_con_d2])
concat_d2_response_period = np.vstack([(response_period_deg.T/np.max(np.abs(response_period_deg[:,:90]),axis = 1)).T,
                                       (response_period_cond.T/np.max(np.abs(response_period_cond[:,:90]),axis = 1)).T])

plot_heatmap_concat(concat_d2,8,line_position,True ,'tracked normalized deg+con d2',
                    vmin=-2,vmax = 2)
#%% kmean

 # kmean
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from sklearn.cluster import KMeans
import sklearn

# X = concat_d2_response_period


# kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
# labels = kmeans_model.labels_
# metrics.calinski_harabaz_score(X, labels)
# k_list = []
# score_list = []
# for k in range(2, 20):
#     kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
#     labels = kmeans_model.labels_
#     print(k, metrics.calinski_harabaz_score(X, labels))
#     k_list.append(k)
#     score_list.append(metrics.calinski_harabaz_score(X, labels))
#     metrics.calinski_harabasz_score(X, labels)
    
# plt.plot(k_list,score_list)
# plt.show()
X = concat_d2_response_period
# auc_mat = np.zeros([X.shape[0],6*5])
# for i in range(6):
#     for j in range(X.shape[0]):
#         auc_mat[j,i*5] = np.trapz(X[j,i*30:i*30+6], dx=0.2)
#         auc_mat[j,i*5+1] = np.trapz(X[j,i*30+6:i*30+12], dx=0.2)
#         auc_mat[j,i*5+2] = np.trapz(X[j,i*30+12:i*30+18], dx=0.2)
#         auc_mat[j,i*5+3] = np.trapz(X[j,i*30+18:i*30+24], dx=0.2)
#         auc_mat[j,i*5+4] = np.trapz(X[j,i*30+24:i*30+30], dx=0.2)

auc_mat = np.zeros([X.shape[0],6*3])
for i in range(6):
    for j in range(X.shape[0]):
        auc_mat[j,i*3] = np.trapz(X[j,i*30:i*30+10], dx=0.2)
        auc_mat[j,i*3+1] = np.trapz(X[j,i*30+10:i*30+20], dx=0.2)
        auc_mat[j,i*3+2] = np.trapz(X[j,i*30+20:i*30+30], dx=0.2)
        
                    

import sklearn
# normalized_array = sklearn.preprocessing.normalize(auc_mat, norm="l1")


kmeans_model = KMeans(n_clusters = 10
                      , random_state=1).fit(np.nan_to_num(auc_mat))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(concat_d2[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'concat')



deg_d2_labels = labels[0:n_mat_deg_d2.shape[0]]

con_d2_labels = labels[n_mat_deg_d2.shape[0]:]

print(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(np.sort(list(Counter(deg_d2_labels).keys())))
print(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(np.sort(list(Counter(con_d2_labels).keys())))



# deg group
sort_index = np.argsort(deg_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_deg_d2[sort_index,:],100,
                            line_position,cum_cluster_num,2, True,
                            'deg')


# con group
sort_index = np.argsort(con_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_con_d2[sort_index,:],100,
                            line_position,cum_cluster_num,2, True,
                            'cond')
#%% plot group avg trace
import itertools
full_X = concat_d2
labels = kmeans_model.labels_
n_clusters = 10
session = [1,3,5,6,8,10]
y_max = [3,  0.6, 1,  1.8,  1, 5,1.4,7, 9, 0.5]
y_min = [-1,-0.4,-1,-0.2,-0.5,-1,-0.4,-1.5,-4,-0.12]

fig,ax = plt.subplots(n_clusters,len(session),figsize = (6.5,1*n_clusters/1.8),sharey = False)
cond_color = itertools.cycle(('#F4CA52','#FFC300',
                         '#FE6B51','#C70039',
                         '#900C3F','#581845'))
deg_color = itertools.cycle(('#F4CA52','#FFC300',
                         '#FE6B51','#18582B',
                         '#528260','#8BAC95'))
for i,cluster in enumerate(set(Counter(labels))):
    index = [i for i, val in enumerate(labels) if val == cluster]
    for j in range(6):
        
        ax[i,j].plot(np.arange(40),np.nanmean(full_X[index,j*40:j*40+40],axis = 0),color = next(deg_color))
        
        ax[i,j].set_ylim([y_min[i],y_max[i]])
        ax[i,j].fill_between(x = [10,15],y1 = 10,y2 = -10,alpha = 0.2,color = 'red',edgecolor = None)
        ax[i,j].axvline(x = 27.5,linewidth=2,alpha = 0.5)
        if i == n_clusters-1 and j != 0:
            sns.despine(left = True,ax = ax[i,j])
            ax[i,j].set_yticks([])
            ax[i,j].set_xticks(np.arange(0,41,5))
            ax[i,j].set_xticklabels(np.arange(0,9,1))
        elif i != n_clusters-1 and j == 0:
            sns.despine(bottom = True,ax = ax[i,j])
            ax[i,j].set_xticks([])
        elif i != n_clusters-1 and j != 0 :
            sns.despine(left = True, bottom = True, ax = ax[i,j])
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
        else:
            ax[i,j].set_xticks(np.arange(0,41,5))
            ax[i,j].set_xticklabels(np.arange(0,9,1))
            sns.despine(ax = ax[i,j])
title = 'avg traces concat d2 10 cluster {}.png'.format(n_clusters)
plt.suptitle(title)
plt.savefig('figures/{}'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
plt.show()

#%% richness of each type
#include D1-05
category_names = ['Type 1', 'Type 2',
                  'Type 3', 'Type 4', 'Type 5','Type 6','Type 7', 'Type 8','Type 9','Type 10']
results = {
       'deg D2': np.around(np.array([8,53,40,6,39,2,29,1,1,74])/np.sum(np.array([8,53,40,6,39,2,29,1,1,74])/100),2),
    'cond D2': np.around(np.array([3,27,27,13,18,4,6,0,0,31])/np.sum(np.array([3,27,27,13,18,4,6,0,0,31])/100),2),
    
}

survey(results, category_names)
plt.savefig('figures/{}'.format('type composition 1 w D1-05 bin 3 6s 10 kmean cluster on normalized'), bbox_inches="tight", dpi = 100,transparent = False)
plt.show()
#%%
df = pd.DataFrame([
                   ['type 1','con d2',300/129],['type 2','con d2',2700/129],['tyep 3','con d2',2700/129],
                   ['type 4','con d2',1300/129],['type 5','con d2',1800/129],['tyep 6','con d2',400/129],
                   ['type 7','con d2',600/129],['type 8','con d2',0/129],['tyep 9','con d2',0/129],
                   ['type 10','con d2',3100/129],
                   ['type 1','deg d2',800/253],['type 2','deg d2',5300/253],['tyep 3','deg d2',4000/253],
                   ['type 4','deg d2',600/253],['type 5','deg d2',3900/253],['tyep 6','deg d2',200/253],
                   ['type 7','deg d2',2900/253],['type 8','deg d2',100/253],['tyep 9','deg d2',100/253],
                   ['type 10','deg d2',7400/253]],columns=['type','group','perc'])
plt.figure(figsize=(7,6))
sns.barplot(data=df, x='type', y='perc', hue='group',palette = ['#FEA572','#70C99A'],)

plt.savefig('figures/{}'.format('type composition 1 barplot bin 3 6s 10 kmean cluster on normalized'))
plt.show()
#%% normalize by the response of each cell's maximum response across all sessions (not deal with cue yet)

#get the maximum absolute response in the odor response period
# deg d1
response_period_deg = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [10,25])

n_mat_deg_d1 = (mat_deg_d1.T/np.max(np.abs(response_period_deg[:,:45]),axis = 1)).T

# plot_heatmap_concat(n_mat_deg_d2,5,[0,0],False,'d2_normalized',vmin = -1.5,vmax = 1.5)
print(n_mat_deg_d1.shape)
# cond d2
response_period_cond = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [10,25])

n_mat_con_d1 = (mat_con_d1.T/np.max(np.abs(response_period_cond[:,:45]),axis = 1)).T
print(n_mat_con_d1.shape)



# concat the two d2 mat
concat_d1 = np.vstack([n_mat_deg_d1,n_mat_con_d1])
concat_d1_response_period = np.vstack([(response_period_deg.T/np.max(np.abs(response_period_deg[:,:45]),axis = 1)).T,
                                       (response_period_cond.T/np.max(np.abs(response_period_cond[:,:45]),axis = 1)).T])
 # kmean
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from sklearn.cluster import KMeans
import sklearn

X = concat_d1_response_period

auc_mat = np.zeros([X.shape[0],6*3])
for i in range(6):
    for j in range(X.shape[0]):
        auc_mat[j,i*3] = np.trapz(X[j,i*15:i*15+5], dx=0.2)
        auc_mat[j,i*3+1] = np.trapz(X[j,i*15+5:i*15+10], dx=0.2)
        auc_mat[j,i*3+2] = np.trapz(X[j,i*15+10:i*15+15], dx=0.2)
        
                    

import sklearn
# normalized_array = sklearn.preprocessing.normalize(auc_mat, norm="l1")


kmeans_model = KMeans(n_clusters = 7
                      , random_state=1).fit(np.nan_to_num(auc_mat))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(concat_d1[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'concat')



deg_d1_labels = labels[0:n_mat_deg_d1.shape[0]]

con_d1_labels = labels[n_mat_deg_d1.shape[0]:]

print(np.array(list(Counter(deg_d1_labels).values()))[np.argsort(list(Counter(deg_d1_labels).keys()))])
print(np.sort(list(Counter(deg_d1_labels).keys())))
print(np.array(list(Counter(con_d1_labels).values()))[np.argsort(list(Counter(con_d1_labels).keys()))])
print(np.sort(list(Counter(con_d1_labels).keys())))



# deg group
sort_index = np.argsort(deg_d1_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(deg_d1_labels).values()))[np.argsort(list(Counter(deg_d1_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_deg_d1[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'deg')


# con group
sort_index = np.argsort(con_d1_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(con_d1_labels).values()))[np.argsort(list(Counter(con_d1_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_con_d1[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'cond')




#%%

# sns.clustermap(np.nan_to_num(concat_d2),cmap = 'RdBu_r',vmin = -1,vmax = 1,row_cluster=True, col_cluster=False )

from scipy.spatial import distance
from scipy.cluster import hierarchy



row_linkage = hierarchy.linkage(
    distance.pdist(concat_d2), method='average')


sns.clustermap(concat_d2, row_linkage=row_linkage, row_cluster=True, col_cluster=False, cmap = 'RdBu_r', method="average",
                figsize=(13, 13))
#%%
def select_active_neurons_index(mat,active_window,response_thres,active_window_thres):
    indexes = []
    for i in range(mat.shape[0]):
        windows = np.zeros([len(active_window),1])
        for k,window in enumerate(active_window):
            windows[k] = 1 if np.mean(mat[i,window[0]:window[1]]) > response_thres else 0
        if sum(windows) > active_window_thres:
            indexes.append(i)
    return np.asarray(indexes)


#%% cosine clustering
from sklearn.cluster import AgglomerativeClustering



index = select_active_neurons_index(mat_deg_d2,[[10,25],[50,65],[90,105],[130,145],[170,185],[210,225]],0.1,3)
sep_deg = mat_deg_d2[index,:].copy()
index = select_active_neurons_index(mat_con_d2,[[10,25],[50,65],[90,105],[130,145],[170,185],[210,225]],0.1,3)
sep_con = mat_con_d2[index,:].copy()
all_mat = np.vstack([sep_deg,sep_con])

# mat for clustering
mat = np.hstack([all_mat[:,10:25],all_mat[:,50:65],all_mat[:,90:105],
                  all_mat[:,130:145],all_mat[:,170:185],all_mat[:,210:225]])
# X = all_mat
# mat = np.zeros([X.shape[0],6*3])
# for i in range(6):
#     for j in range(X.shape[0]):
#         mat[j,i*3] = np.trapz(X[j,i*15:i*15+5], dx=0.2)
#         mat[j,i*3+1] = np.trapz(X[j,i*15+5:i*15+10], dx=0.2)
#         mat[j,i*3+2] = np.trapz(X[j,i*15+10:i*15+15], dx=0.2)

# X = all_mat
# mat = np.zeros([X.shape[0],6])
# for i in range(6):
#     for j in range(X.shape[0]):
#         mat[j,i] = np.mean(X[j,i*15:i*15+15])
        
cluster = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage = 'average').fit(mat)
labels = cluster.labels_
sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(all_mat[sort_index,:],8,
                            line_position,cum_cluster_num,8, False,
                            'concat')


deg_d2_labels = labels[0:sep_deg.shape[0]]

con_d2_labels = labels[sep_deg.shape[0]:]

print(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(np.sort(list(Counter(deg_d2_labels).keys())))
print(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(np.sort(list(Counter(con_d2_labels).keys())))



# deg group
sort_index = np.argsort(deg_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(sep_deg[sort_index,:],8,
                            line_position,cum_cluster_num,5, False,
                            'deg')


# con group
sort_index = np.argsort(con_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(sep_con[sort_index,:],8,
                            line_position,cum_cluster_num,5, False,
                            'cond')

#%% h vs kmeans
kmeans_model = KMeans(n_clusters = 12, random_state=1).fit(np.nan_to_num(concat_d2))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(concat_d2[sort_index,:],100,
                            line_position,cum_cluster_num,1, False,
                            'concat')

#%% clustering based on correlation
import pandas as pd
import scipy.cluster.hierarchy as spc


df = pd.DataFrame(my_data)
corr = df.corr().values

pdist = spc.distance.pdist(corr)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')   

#%% time-warping method
import tslearn
from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=15, metric="dtw", max_iter=5)
model.fit(np.nan_to_num(concat_d2_response_period))


labels = model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(concat_d2[sort_index,:],100,
                            line_position,cum_cluster_num,1, False,
                            'concat')


deg_d2_labels = labels[0:n_mat_deg_d2.shape[0]]

con_d2_labels = labels[n_mat_deg_d2.shape[0]:]

print(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(np.sort(list(Counter(deg_d2_labels).keys())))
print(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(np.sort(list(Counter(con_d2_labels).keys())))



# deg group
sort_index = np.argsort(deg_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(deg_d2_labels).values()))[np.argsort(list(Counter(deg_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_deg_d2[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'deg')


# con group
sort_index = np.argsort(con_d2_labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(con_d2_labels).values()))[np.argsort(list(Counter(con_d2_labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*40)
    line_position.append(15+i*40)
    line_position.append(27.5+i*40)
plot_heatmap_concat_nonsort(n_mat_con_d2[sort_index,:],100,
                            line_position,cum_cluster_num,2, False,
                            'cond')
    
    
    
    
    
    
    
    
    
    
    
    