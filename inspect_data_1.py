#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:30:10 2021

@author: lechenqian

purpose: 
    1. across trial variations
    2. go omit, go odor trial's responses' consistency
"""

#%% import
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

# %% functions
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


def plot_heatmap2(avg_mat,odor_on_frame,window_pre,window_post,line_position,is_save,title):
    
    
    
    
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
    

def plot_heatmap_concat(avg_mat,window_pre,line_position,is_save,title,vmin=-5,vmax = 5,figsize = (12,7),x_ticks = list(np.arange(8))):
    
    
    plt.figure(figsize = figsize)
    
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
    ticks = list(x_ticks) * int(len(line_position)/3)
    plt.xticks(np.arange(0, 5*len(x_ticks)*int(len(line_position)/3), step=5),ticks)
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
    
#%% LOAD DATA

filename ="/Users/lechenqian/Dropbox (Uchida Lab)/Analyze_imaging_data/partial_to_mac_2021-11-03_avg_trace_registered_nan_filled_added.pickle"
partial_dict = load_pickleddata(filename)
# %%% define mouse groups
# deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d1 = ['D1-05','D1-02','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']
deg_d1_notr  = ['D1-01','D1-03','D1-09'] # NOT BEING REGISTERED

# %%% concate signals from 6 sessions horizontally and stack go,no go on the third dimension

mat_deg_d2_go = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d2_nogo = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])

mat_con_d2_go = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])

mat_con_d2_nogo = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])

concat_go_nogo_deg_d2 = np.dstack([mat_deg_d2_go,mat_deg_d2_nogo])
concat_go_nogo_con_d2 = np.dstack([mat_con_d2_go,mat_con_d2_nogo])

# %%% concate signals from 6 sessions horizontally and stack go,go omission,no go horizontally


mat_deg_d2_go = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d2_go_omit = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])
mat_deg_d2_nogo = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])

mat_con_d2_go = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_con_d2_go_omit = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])

mat_con_d2_nogo = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])

mat_deg_d1_go = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d1_go_omit = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])
mat_deg_d1_nogo = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])



mat_con_d1_go = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_con_d1_go_omit = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])

mat_con_d1_nogo = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'no_go','axis1',slice_range = [0,40])

concat_go_omit_nogo_deg_d2 = np.hstack([mat_deg_d2_go,mat_deg_d2_go_omit,mat_deg_d2_nogo])
concat_go_omit_nogo_con_d2 = np.hstack([mat_con_d2_go,mat_con_d2_go_omit,mat_con_d2_nogo])
concat_go_omit_nogo_con_d1 = np.hstack([mat_con_d1_go,mat_con_d1_go_omit,mat_con_d1_nogo])
concat_go_omit_nogo_deg_d1 = np.hstack([mat_deg_d1_go,mat_deg_d1_go_omit,mat_deg_d1_nogo])

# %%% concate signals from 6 sessions horizontally and stack go,go omission,no go horizontally


mat_deg_d2_go = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d2_go_omit = stack_sliced_avgtraces_from_multi_mice(deg_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])


mat_con_d2_go = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_con_d2_go_omit = stack_sliced_avgtraces_from_multi_mice(con_d2, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])

mat_deg_d1_go = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_deg_d1_go_omit = stack_sliced_avgtraces_from_multi_mice(deg_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])

mat_con_d1_go = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go','axis1',slice_range = [0,40])
mat_con_d1_go_omit = stack_sliced_avgtraces_from_multi_mice(con_d1, partial_dict,
                                              'registered_aligned_dff_average', 
                                              'go_omit','axis1',slice_range = [0,40])


concat_go_omit_avg_deg_d2 = np.hstack([mat_deg_d2_go,mat_deg_d2_go_omit,(mat_deg_d2_go+mat_deg_d2_go_omit)/2])
concat_go_omit_avg_con_d2 = np.hstack([mat_con_d2_go,mat_con_d2_go_omit,(mat_con_d2_go+mat_con_d2_go_omit)/2])
concat_go_omit_avg_deg_d1 = np.hstack([mat_deg_d1_go,mat_deg_d1_go_omit,(mat_deg_d1_go+mat_deg_d1_go_omit)/2])
concat_go_omit_avg_con_d1 = np.hstack([mat_con_d1_go,mat_con_d1_go_omit,(mat_con_d1_go+mat_con_d1_go_omit)/2])



#%%% visualize it go nogo goomit
avg_mat = concat_go_omit_nogo_deg_d1
window_pre = 40*17+12
line_position = [10+40*i for i in range(6*3)]+[15+40*i for i in range(6*3)]+[27.5+40*i for i in range(6*3)]
is_save = False
title = 'go go omit no go'
plot_heatmap_concat(avg_mat,window_pre,line_position,
                    is_save,title,vmin=-8,vmax = 8,
                    figsize = (20,7),x_ticks = list(np.arange(8)))
#%%% visualzie difference between go and avg of go and go_omit
avg_mat = concat_go_omit_avg_con_d1
window_pre = 40*17+10
line_position = [10+40*i for i in range(6*3)]+[15+40*i for i in range(6*3)]+[27.5+40*i for i in range(6*3)]
is_save = False
title = 'go go omit no go'
plot_heatmap_concat(avg_mat,window_pre,line_position,
                    is_save,title,vmin=-7,vmax = 7,
                    figsize = (20,7),x_ticks = list(np.arange(8)))

#%% inspect the trial variations
data = partial_dict['D2-04']['aligned_dff_multitrials']
for session in range(6):
    
    data_type_session = data['go'][session]
    fig,axes = plt.subplots(5,int(np.ceil(data_type_session.shape[0]/5)),figsize = (30,15))
    for i in range(5):
        for j in range(int(np.ceil(data_type_session.shape[0]/5))):
            try:
                sns.heatmap(data_type_session[j*5+i,:,:].T,vmin = -5,vmax = 5,ax = axes[i,j],cmap = 'RdYlBu')
            except: 
                pass
    plt.show()






































































    
    
    
    
    
    