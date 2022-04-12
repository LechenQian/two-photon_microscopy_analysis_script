#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:23:28 2021

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
filename ="/Users/lechenqian/Dropbox (Uchida Lab)/Analyze_imaging_data/partial_to_mac_2021-10-23_registration_and_registerd_matrices_added.pickle.pickle"
partial_dict = load_pickleddata(filename)
#%%
# deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d1 = ['D1-05','D1-02','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']
deg_d1_notr  = ['D1-01','D1-03','D1-09']
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

def plot_heatmap_concat(avg_mat,window_pre,line_position,is_save,title):
    
    
    plt.figure(figsize = (12,7))
    
    if window_pre == 5:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+15)],axis = 1)
    else:
        peak = np.mean(avg_mat[:,int(window_pre):int(window_pre+15)],axis = 1)
    order = np.argsort(-peak)
    sns.heatmap(avg_mat[order,:],vmin = -5,vmax = 5,cmap = 'RdBu_r')
    for pos in line_position:
        plt.axvline(x = int(np.ceil(pos)),color = 'grey')
    
    plt.ylabel('Neuron #')
    plt.xlabel('Time(s)')
    ticks = [0,1,2,3,4,5,6,7,8] * 6
    plt.xticks(np.arange(0, 5*9*6, step=5),ticks)
    plt.yticks(np.arange(0, avg_mat.shape[0]+1, step=50),np.arange(0, avg_mat.shape[0]+1, step=50))
    plt.title(title)
    if is_save:
        plt.savefig('figures/{}.png'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
    plt.show()
    
def plot_heatmap_concat_nonsort(avg_mat,window_pre,line_position,h_line_position,vlim, is_save,title):
    
    
    plt.figure(figsize = (12,7))

    sns.heatmap(avg_mat,vmin = -vlim,vmax = vlim,cmap = 'RdBu_r')
    for pos in line_position:
        plt.axvline(x = int(np.ceil(pos)),color = 'grey')
    for pos in h_line_position:
        plt.axhline(y = int(np.ceil(pos)),color = 'cyan')
    
    plt.ylabel('Neuron #')
    plt.xlabel('Time(s)')
    ticks = [0,1,2,3,4,5,6,7,8] * 6
    plt.xticks(np.arange(0, 5*9*6, step=5),ticks)
    plt.yticks(np.arange(0, avg_mat.shape[0]+1, step=50),np.arange(0, avg_mat.shape[0]+1, step=50))
    plt.title(title)
    if is_save:
        plt.savefig('figures/{}.png'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
    plt.show()
#%%
session = [1,3,5,6,8,10]
for i in range(6):
    deg_D1_go = stack_traces_from_multi_mice(deg_d1+deg_d1_notr, partial_dict,
                                                  'aligned_dff_average', 'go','axis1',
                                                  partial = True, index_list=[i])
    # sns.heatmap(deg_D1_go_cond1)
    
    
    plot_heatmap2(deg_D1_go,10,10,50,[10,15,27.5],True,'deg d1 session {} W D1-05'.format(session[i]))
#%% plot psth
import itertools
session = [1,3,5,6,8,10]
color = itertools.cycle(('#FBDA7A','#F99246',
                         '#B12D25','#3325B1',
                         '#255DB1','#25A3B1'))
for i in range(6):
    deg_D1_go = stack_traces_from_multi_mice(deg_d1, partial_dict,
                                                  'aligned_dff_average', 'go','axis1',
                                                  partial = True, index_list=[i])
    # sns.heatmap(deg_D1_go_cond1)
    ACTIVE = deg_D1_go[np.sum(deg_D1_go[:,13:18],axis = 1)>2,:]
    INHIBIT = deg_D1_go[np.sum(deg_D1_go[:,17:21],axis = 1)<-0.6,:]
    current_color = next(color)
    plt.plot(np.arange(ACTIVE.shape[1]),np.nanmean(ACTIVE,axis = 0),label = 'session {}'.format(session[i]),color = current_color)
    plt.plot(np.arange(INHIBIT.shape[1]),np.nanmean(INHIBIT,axis = 0),color = current_color)
    


plt.legend()

plt.title('deg D1')
plt.xticks(np.arange(0, 46, step=5),[0,1,2,3,4,5,6,7,8,9])
sns.despine()
plt.savefig('figures/deg d1 active inhibit W D1-05 wo notr.png', bbox_inches="tight", dpi = 100,transparent = True)
   
plt.show()
#%%
session = [1,3,5,6,8,10]
for i in range(6):
    deg_D1_go = stack_traces_from_multi_mice(con_d2, partial_dict,
                                                  'aligned_dff_average', 'go','axis1',
                                                  partial = True, index_list=[i])
    # sns.heatmap(deg_D1_go_cond1)
    
    
    plot_heatmap2(deg_D1_go,10,10,50,[10,15,27.5],True,'cond d2 session {}'.format(session[i]))

#%%

deg_d1 = ['D1-02','D1-12','D1-13']
#%%
mat = stack_traces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
# sns.heatmap(deg_D1_go_cond1)

line_position = []
for i in range(6):
    line_position.append(10+i*45)
    line_position.append(15+i*45)
    line_position.append(27.5+i*45)
plot_heatmap_concat(mat,100,line_position,True,'tracked deg d1 w d1-05')
#%%
import sklearn
normalized_mat = sklearn.preprocessing.normalize(mat, norm="l1")
plot_heatmap_concat(normalized_mat*200,10,line_position,True,'tracked deg d1 w d1-05 normed')
#%%
deg_d1 = ['D1-05']
mat = stack_traces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
# sns.heatmap(deg_D1_go_cond1)

line_position = []
for i in range(6):
    line_position.append(10+i*45)
    line_position.append(15+i*45)
    line_position.append(27.5+i*45)
plot_heatmap_concat(mat,100,line_position,True,'tracked D1-05')

#%% kmeans parts
mat_deg_d1_full = stack_traces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
mat_deg_d2_full = stack_traces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
mat_con_d1_full = stack_traces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
mat_con_d2_full = stack_traces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              partial = False, index_list=None)
# sns.clustermap(np.nan_to_num(mat_deg_d1),cmap = 'RdBu_r',vmin = -5,vmax = 5,row_cluster=True, col_cluster=False )




# select odor on to 4s after odor on and concat the registered trials

mat_deg_d1 = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [10,40])
mat_deg_d2 = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [10,40])
mat_con_d1 = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [10,40])
mat_con_d2 = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 'go','axis1',
                                              slice_range = [10,40])
#%%
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from sklearn.cluster import KMeans
import sklearn
#%%
X = mat_deg_d2


kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)
k_list = []
score_list = []
for k in range(2, 20):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    print(k, metrics.calinski_harabaz_score(X, labels))
    k_list.append(k)
    score_list.append(metrics.calinski_harabaz_score(X, labels))
    metrics.calinski_harabasz_score(X, labels)
    
plt.plot(k_list,score_list)
plt.show()
#%%
X = mat_deg_d2
kmeans_model = KMeans(n_clusters=10, random_state=1).fit(np.nan_to_num(X))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*45)
    line_position.append(15+i*45)
    line_position.append(27.5+i*45)
plot_heatmap_concat_nonsort(mat_deg_d2_full[sort_index,:],100,
                            line_position,cum_cluster_num, True,
                            'tracked deg d2 kmean10.png')




#%% area under curve
X = mat_deg_d1
full_X = mat_deg_d1_full
auc_mat = np.zeros([X.shape[0],6*5])
for i in range(6):
    for j in range(X.shape[0]):
        auc_mat[j,i*5] = np.trapz(X[j,i*30:i*30+6], dx=0.2)
        auc_mat[j,i*5+1] = np.trapz(X[j,i*30+6:i*30+12], dx=0.2)
        auc_mat[j,i*5+2] = np.trapz(X[j,i*30+12:i*30+18], dx=0.2)
        auc_mat[j,i*5+3] = np.trapz(X[j,i*30+18:i*30+24], dx=0.2)
        auc_mat[j,i*5+4] = np.trapz(X[j,i*30+24:i*30+30], dx=0.2)
                    

import sklearn
normalized_array = sklearn.preprocessing.normalize(auc_mat, norm="l1")
kmeans_model = KMeans(n_clusters=9, random_state=1).fit(np.nan_to_num(normalized_array))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*45)
    line_position.append(15+i*45)
    line_position.append(27.5+i*45)
plot_heatmap_concat_nonsort(full_X[sort_index,:],100,
                            line_position,cum_cluster_num,8, True,
                            'tracked deg d1 w d1-05 norm auc kmean 9.png')
#%% rerun inhibition neurons
X = mat_con_d2
full_X = mat_con_d2_full
auc_mat = np.zeros([X.shape[0],6*5])
for i in range(6):
    for j in range(X.shape[0]):
        auc_mat[j,i*5] = np.trapz(X[j,i*30:i*30+6], dx=0.2)
        auc_mat[j,i*5+1] = np.trapz(X[j,i*30+6:i*30+12], dx=0.2)
        auc_mat[j,i*5+2] = np.trapz(X[j,i*30+12:i*30+18], dx=0.2)
        auc_mat[j,i*5+3] = np.trapz(X[j,i*30+18:i*30+24], dx=0.2)
        auc_mat[j,i*5+4] = np.trapz(X[j,i*30+24:i*30+30], dx=0.2)
                    

normalized_array = sklearn.preprocessing.normalize(auc_mat, norm="l1")
kmeans_model = KMeans(n_clusters=5, random_state=1).fit(np.nan_to_num(normalized_array))

p_labels = kmeans_model.labels_.copy()
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(np.nan_to_num(normalized_array[p_labels==1,:]))
labels = kmeans_model.labels_

sort_index = np.argsort(labels)

cum_cluster_num = np.cumsum(np.array(list(Counter(labels).values()))[np.argsort(list(Counter(labels).keys()))])
print(cum_cluster_num)

line_position = []
for i in range(6):
    line_position.append(10+i*45)
    line_position.append(15+i*45)
    line_position.append(27.5+i*45)
plot_heatmap_concat_nonsort(full_X[p_labels == 1,:][sort_index,:],100,
                            line_position,cum_cluster_num, False,
                            'tracked con d1 norm auc kmean10.png')
#%%
X = normalized_array
k_list = []
score_list = []
for k in range(2, 20):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    print(k, metrics.calinski_harabaz_score(X, labels))
    k_list.append(k)
    score_list.append(metrics.calinski_harabaz_score(X, labels))
    metrics.calinski_harabasz_score(X, labels)
    
plt.plot(k_list,score_list)
plt.show()


#%%
import itertools
session = [1,3,5,6,8,10]
X = mat_deg_d1
full_X = mat_deg_d1_full
auc_mat = np.zeros([X.shape[0],6*5])
for i in range(6):
    for j in range(X.shape[0]):
        auc_mat[j,i*5] = np.trapz(X[j,i*30:i*30+6], dx=0.2)
        auc_mat[j,i*5+1] = np.trapz(X[j,i*30+6:i*30+12], dx=0.2)
        auc_mat[j,i*5+2] = np.trapz(X[j,i*30+12:i*30+18], dx=0.2)
        auc_mat[j,i*5+3] = np.trapz(X[j,i*30+18:i*30+24], dx=0.2)
        auc_mat[j,i*5+4] = np.trapz(X[j,i*30+24:i*30+30], dx=0.2)
                    


n_clusters = 9
normalized_array = sklearn.preprocessing.normalize(auc_mat, norm="l1")
kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(np.nan_to_num(normalized_array))
labels = kmeans_model.labels_
print(Counter(labels).keys())
print(Counter(labels).values())
sort_index = np.argsort(labels)

fig,ax = plt.subplots(n_clusters,len(session),figsize = (8,1*n_clusters))
cond_color = itertools.cycle(('#F4CA52','#FFC300',
                         '#FE6B51','#C70039',
                         '#900C3F','#581845'))
deg_color = itertools.cycle(('#F4CA52','#FFC300',
                         '#FE6B51','#18582B',
                         '#528260','#8BAC95'))
for i,cluster in enumerate(set(Counter(labels))):
    index = [i for i, val in enumerate(labels) if val == cluster]
    for j in range(6):
        
        ax[i,j].plot(np.arange(45),np.nanmean(full_X[index,j*45:j*45+45],axis = 0),color = next(deg_color))
        ax[i,j].set_ylim([-4.5,9.5])
        ax[i,j].fill_between(x = [10,15],y1 = 10,y2 = -10,alpha = 0.2,color = 'red',edgecolor = None)
        ax[i,j].axvline(x = 27.5,linewidth=2,alpha = 0.5)
        if i == n_clusters-1 and j != 0:
            sns.despine(left = True,ax = ax[i,j])
            ax[i,j].set_yticks([])
            ax[i,j].set_xticks(np.arange(0,46,5))
            ax[i,j].set_xticklabels(np.arange(0,10,1))
        elif i != n_clusters-1 and j == 0:
            sns.despine(bottom = True,ax = ax[i,j])
            ax[i,j].set_xticks([])
        elif i != n_clusters-1 and j != 0 :
            sns.despine(left = True, bottom = True, ax = ax[i,j])
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
        else:
            ax[i,j].set_xticks(np.arange(0,46,5))
            ax[i,j].set_xticklabels(np.arange(0,10,1))
            sns.despine(ax = ax[i,j])
title = 'avg traces deg d1 w D1-05 cluster {}.png'.format(n_clusters)
plt.suptitle(title)
plt.savefig('figures/{}'.format(title), bbox_inches="tight", dpi = 100,transparent = True)
plt.show()

#%%
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
    category_colors = plt.get_cmap('RdYlGn')(
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



#%%
category_names = ['Type 1', 'Type 2',
                  'Type 3', 'Type 4', 'Type 5','Type 6']
results = {
    'deg D1': np.around(np.array([11, 15,10, 16, 0,35])/np.sum(np.array([11, 15,10, 16, 0,35])/100),2),
    'cond D1': np.around(np.array([4, 4, 13, 9, 3, 7])/np.sum(np.array([4, 4, 13, 9, 3, 7])/100),2),
    'deg D2': np.around(np.array([25+24, 30+22+24, 10, 78,0, 14])/np.sum(np.array([25+24, 30+22+24, 10, 78,0, 14])/100),2),
    'cond D2': np.around(np.array([29, 14,35, 12,0, 39])/np.sum(np.array([29, 14, 35, 12,0, 39])/100),2),
    
}

survey(results, category_names)
plt.show()
#%%
category_names = ['drop aft 5', 'drop aft 1','drop aft 3',
                  'drop aft 6', 'drop aft 8', 'increase']
results = {
    'deg D1': np.around(np.array([15, 0,15, 0, 0,11])/np.sum(np.array([15, 0,15, 0, 0,11])/100),2),
    'cond D1': np.around(np.array([4, 0,4,10,4,3])/np.sum(np.array([4, 0,4,10,4,3])/100),2),
    'deg D2': np.around(np.array([30+5, 23, 22, 17,3, 15+24+5])/np.sum(np.array([30+5, 23, 22, 17,3, 15+24+5])/100),2),
    'cond D2': np.around(np.array([14, 0,19, 12,0, 29])/np.sum(np.array([14, 0,19, 12,0, 29])/100),2),
    
}

survey(results, category_names)
plt.show()
#%% include D1-05
category_names = ['Type 1', 'Type 2',
                  'Type 3', 'Type 4', 'Type 5','Type 6']
results = {
    'deg D1': np.around(np.array([39, 47,43, 24+13, 0,23])/np.sum(np.array([39, 47,43, 24+13, 0,23])/100),2),
    'cond D1': np.around(np.array([4, 4, 13, 9, 3, 7])/np.sum(np.array([4, 4, 13, 9, 3, 7])/100),2),
    'deg D2': np.around(np.array([25+24, 30+22+24, 10, 78,0, 14])/np.sum(np.array([25+24, 30+22+24, 10, 78,0, 14])/100),2),
    'cond D2': np.around(np.array([29, 14,35, 12,0, 39])/np.sum(np.array([29, 14, 35, 12,0, 39])/100),2),
    
}

survey(results, category_names)
plt.savefig('figures/{}'.format('type composition 1 w D1-05'), bbox_inches="tight", dpi = 100,transparent = True)
plt.show()
#%% TYPE 2 include D1-05
category_names = ['drop aft 5', 'drop aft 1','drop aft 3',
                  'drop aft 6', 'drop aft 8', 'increase']
results = {
    'deg D1': np.around(np.array([26, 9,24, 5, 0,30])/np.sum(np.array([26, 9,24, 5, 0,30])/100),2),
    'cond D1': np.around(np.array([4, 0,4,10,4,3])/np.sum(np.array([4, 0,4,10,4,3])/100),2),
    'deg D2': np.around(np.array([30+5, 23, 22, 17,3, 15+24+5])/np.sum(np.array([30+5, 23, 22, 17,3, 15+24+5])/100),2),
    'cond D2': np.around(np.array([14, 0,19, 12,0, 29])/np.sum(np.array([14, 0,19, 12,0, 29])/100),2),
    
}

survey(results, category_names)
plt.savefig('figures/{}'.format('type composition 2 w D1-05'), bbox_inches="tight", dpi = 100,transparent = True)
plt.show()