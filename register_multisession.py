# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:32:22 2021

@author: qianl
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle5
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import watershed, dilation
from skimage.segmentation import find_boundaries
from skimage.io import imsave, imread
from skimage.feature import peak_local_max
from skimage.external import tifffile as tif
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


def stack_traces_from_multi_mice(mice_name:list, data:dict, keyname :str, trialtype: str):
    traces_tensor_list = [] 
    partial = True if input('traces registered? (y/n)>>  ') == 'n' else False
    index_list = input('enter a list of indexes you want to concat:>>  ')
    index_list = json.loads(index_list)
    for mouse in mice_name:
        if not partial:
            temp_list = [item for item in data[mouse][keyname][trialtype]]
            stacked_multisesions_traces = np.concatenate(temp_list,axis = 2)
        else:
            
            temp_list = [data[mouse][keyname][trialtype][i] for i in index_list]
            try:
                stacked_multisesions_traces = np.concatenate(temp_list,axis = 2)
            except:
                stacked_multisesions_traces = temp_list.copy()
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


filename ="D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled/from_mice_pickle_and_hdf5_loaded_processed_2021-10-13_registration_added.pickle.pickle"
big_dict = load_pickleddata(filename)
#%%
deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']


for mouse in deg_d1+deg_d2+con_d1+con_d2:
    
    assignment = big_dict[mouse]['registration_mat']
    print(mouse)
    
    n_reg = 6
    alignments = filter_assignment(assignment, n_reg)
    
    big_dict[mouse]['registered_aligned_dff_multitrials'] = {'go':[],'no_go':[],'go_omit': [],'unpred_water':[]}
    big_dict[mouse]['registered_aligned_dff_average'] = {'go':[],'no_go':[],'go_omit': [],'unpred_water':[]}
    for i in range(n_reg):
        
        for ttype in getKeys(big_dict[mouse]['aligned_dff_multitrials']):
            if ttype == 'unpred_water':
                
                if i>2:
                    try:
                        new_mat = big_dict[mouse]['aligned_dff_multitrials'][ttype][i-3][alignments[:, i], :, :]
                        big_dict[mouse]['registered_aligned_dff_multitrials'][ttype].append(new_mat)
                        new_mat2 = big_dict[mouse]['aligned_dff_average'][ttype][i-3][alignments[:, i], :]
                        big_dict[mouse]['registered_aligned_dff_average'][ttype].append(new_mat2)
                    except:
                        pass
            elif ttype in ['go','go_omit','no_go']:
                try:
                    new_mat = big_dict[mouse]['aligned_dff_multitrials'][ttype][i][alignments[:, i], :, :]
                    big_dict[mouse]['registered_aligned_dff_multitrials'][ttype].append(new_mat)
                    new_mat2 = big_dict[mouse]['aligned_dff_average'][ttype][i][alignments[:, i], :]
                    big_dict[mouse]['registered_aligned_dff_average'][ttype].append(new_mat2)
                except:
                    pass
        

#%%%
path = "D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled"
filename = "from_mice_pickle_and_hdf5_loaded_processed_2021-10-13_registration_and_registerd_matrices_added.pickle"
pickle_dict(big_dict, path, filename)


#%% pickle part of the big_dict
deg_d1_notr = ['D1-01','D1-03','D1-09']
deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']

partial_dict_tomac = {}
partial_dict_tomac['INFO'] = {'deg_D1':deg_d1,'deg_D2':deg_d2,'con_D1':con_d1,'con_D2':con_d2,'deg_D1_notregistered':deg_d1_notr}

for mouse in deg_d1+deg_d2+con_d1+con_d2+deg_d1_notr:
    partial_dict_tomac[mouse] = {}
for mouse in deg_d1+deg_d2+con_d1+con_d2+deg_d1_notr:
    try:
        partial_dict_tomac[mouse]['registered_aligned_dff_multitrials'] = big_dict[mouse]['registered_aligned_dff_multitrials']
    except:
        pass
    try:
        partial_dict_tomac[mouse]['registered_aligned_dff_average'] = big_dict[mouse]['registered_aligned_dff_average']
    except:
        pass
    try:
        partial_dict_tomac[mouse]['aligned_dff_multitrials'] = big_dict[mouse]['aligned_dff_multitrials']
    except:
        pass
    try:
        partial_dict_tomac[mouse]['aligned_dff_average'] = big_dict[mouse]['aligned_dff_average']
    except:
        pass
    if mouse in deg_d1+deg_d1_notr:
        partial_dict_tomac[mouse]['group'] = 'deg_D1'
    elif mouse in deg_d2:
        partial_dict_tomac[mouse]['group'] = 'deg_D2'
    elif mouse in con_d1:
        partial_dict_tomac[mouse]['group'] = 'con_D1'
    elif mouse in con_d2:
        partial_dict_tomac[mouse]['group'] = 'con_D2'
        
    partial_dict_tomac[mouse]['session_id'] = big_dict[mouse]['session_id']
    try:
        partial_dict_tomac[mouse]['registration_mat'] = big_dict[mouse]['registration_mat']
    except:
        pass
    partial_dict_tomac[mouse]['spatial_footprints'] = big_dict[mouse]['spatial_footprints']
    partial_dict_tomac[mouse]['normed_templates'] = big_dict[mouse]['normed_templates']
#%%

print(getKeys(partial_dict_tomac))
print(getKeys(partial_dict_tomac['D1-01']))
print(getKeys(partial_dict_tomac['D1-02']))
print(getKeys(partial_dict_tomac['D1-15']))


path = "D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled"
filename = "partial_to_mac_2021-10-23_registration_and_registerd_matrices_added.pickle"
pickle_dict(partial_dict_tomac, path, filename)











#%%
group_concat_dict = {}
group_concat_dict['degD1'] = stack_traces_from_multi_mice(deg_d1,big_dict,'aligned_dff_multitrials','go')

print(group_concat_dict['degD1'].shape)

# #%%
# Activities = []
# for index, mat in enumerate(big_dict['D1-05']['aligned_dff_multitrials']['go']):
#     for trial in range(mat.shape[2]):
#         x = mat[:,5:20,trial]
#         peak_index = np.argmax(abs(x),axis = 1)
#         peak = [x[i,peak] for i,peak in enumerate(peak_index)]
        
#         Activities += peak

# fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# # We can set the number of bins with the `bins` kwarg
# axs.hist(Activities, bins=40)
# plt.title('hist of peak absolute activities of D1-05 from all sessions')
# plt.xlabel('zscore')
# plt.ylabel('counts')
# plt.show()

# mat = big_dict['D1-05']['aligned_dff_multitrials']['go'][2]
# for trial in range(mat.shape[2]):
#     x = mat[:,5:20,trial]
#     peak_index = np.argmax(abs(x),axis = 1)
#     peak = [x[i,peak] for i,peak in enumerate(peak_index)]
    
#     Activities += peak

# fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# # We can set the number of bins with the `bins` kwarg
# axs.hist(Activities, bins=20)
# plt.title('hist of peak absolute activities of D1-05 from session 5')
# plt.xlabel('zscore')
# plt.ylabel('counts')
# plt.show()









