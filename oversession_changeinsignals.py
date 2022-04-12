# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:47:23 2020

@author: qianl
"""

#%% import
import pickle
from caiman.base.rois import register_multisession
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.source_extraction.cnmf import params as params
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from caiman.utils.utils import load_dict_from_hdf5
from caiman.utils.utils import recursively_load_dict_contents_from_group
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import sys
from skimage import io
import time
import h5py
import os
import math
import cv2
import re
import scipy
from matplotlib.cm import get_cmap
import seaborn as sns
import bokeh.plotting as bpl
import matplotlib as mpl
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2

from glob import glob

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils import visualization
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.utils.utils import load_dict_from_hdf5
import holoviews as hv
bpl.output_notebook()
hv.notebook_extension('bokeh')
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
#%% functions
def pickle_dict(df,path,filename):
    try:
        os.makedirs(path) # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path +'/{}.pickle'.format(filename)
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save to pickle done!')


def load_pickleddata(filename):
    
    with open(filename, 'rb') as handle:
        df = pickle.load(handle)
    return df
def getKeys(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
    return list
def read_filenames_in_folder(filedir,postfix):
    
    filenames = []
    file_index = []
    for dirpath, dirnames, files in os.walk(filedir): # can walk through all levels down
        #     print(f'Found directory: {dirpath}')
        for f_name in files:
            if f_name.endswith(postfix):
                filename = dirpath+'/'+f_name
                try:
                    index = list(map(int,re.findall(r"(\d{4,5})",f_name)))[0]
                    filenames.append(filename)
                    file_index.append(index)
                except:
                    pass
    reindex = np.argsort(file_index)
    sort_file_index = np.sort(file_index)
    
    sort_filenames = [filenames[i] for i in reindex]
    print('---------------------------------------------')    
    print('Files have been loaded.')
    return sort_filenames
def load_files_into_dict(sort_filenames):
    Dicts = []
    for filename in sort_filenames:
        if filename.endswith('.pickle'):
            big_dict = load_pickleddata(filename)
            Dicts.append(big_dict)
            print(filename)
        elif filename.endswith('.hdf5'):
            cnmf = load_dict_from_hdf5(filename)
            Dicts.append(cnmf)
            print(filename)
    return Dicts
def slice_signal_to_trial(temp_traces,df): 
    # signal matrix: neurons * temporal traces; frames: the colomn for frame length, df.frame
    F_dff_individualtrials = {}
    cum_frames = np.cumsum(df.frames.values)
    cum_frames = np.insert(cum_frames, 0,0)
    for i in range(len(cum_frames)-1):
        F_dff_individualtrials['Trial'+str(i)] = temp_traces[:,cum_frames[i]:cum_frames[i+1]].copy()
        
    return F_dff_individualtrials

def find_align_position(framenum_trialtype,frameswitch_trialtype):
    pre = []
    post = []
    for i in range(len(framenum_trialtype)):
        pre_temp = frameswitch_trialtype[i][0]
        post_temp = framenum_trialtype[i] - pre_temp
        pre.append(pre_temp)
        post.append(post_temp)
    max_pre = np.max(pre)
    max_post = np.max(post)
    len_nan = int(max_pre+max_post+1)
    return max_pre, max_post,len_nan

def subtract_baseline(stacked_matrix,soleopen_frame):
    
    sub = stacked_matrix.copy()
    for i in range(stacked_matrix.shape[0]):
        value = sub[i,0:int(soleopen_frame)]
        
        if len(value) >1:
            avg = value.mean()
            
        else:
            avg = 0
        
        sub[i,:] =  sub[i,:] - avg
        
    return sub

def create_3d_matrix(num_neurons, len_nan, num_go, odor_on_frame, temp_traces, 
                     index_trialtype, framenum_trialtype,frameswitch_trialtype,neuron_baseline_std): 
    ### for the xxx_trialtype, the type has to be specified when give inputs
    a = np.full([num_neurons, len_nan, num_go], np.nan)
    
    for i, index in enumerate(index_trialtype):
        start_p = int(odor_on_frame-frameswitch_trialtype[i][0])
        end_p = int(start_p + framenum_trialtype[i])
        sub = subtract_baseline(temp_traces['Trial{}'.format(index)], frameswitch_trialtype[i][0])
        zscore = calculate_zscore(sub,neuron_baseline_std)
        a[:,start_p:end_p,i] = zscore
        
    return a
def std_in_baseline(num_neurons,bg_trialtype,temp_traces,dict_index_trialtype):
    neuron_lump_bgsignal = []
    num_neurons = temp_traces['Trial1'].shape[0]
    
    index_bg = dict_index_trialtype[bg_trialtype] 
    for i, index in enumerate(index_bg):
        for j in range(num_neurons):
            if i == 0 :
                neuron_lump_bgsignal.append(temp_traces['Trial{}'.format(index)][j,:].tolist())
            else:
                neuron_lump_bgsignal[j].append(temp_traces['Trial{}'.format(index)][j,:].tolist()[0])
    neuron_baseline_std = np.std(np.asarray(neuron_lump_bgsignal),axis = 1)       
    return neuron_baseline_std
def calculate_zscore(sub,neuron_baseline_std):
    zscore = sub.T/neuron_baseline_std.T
    return zscore.T

def genereate_dff_aligned_matrix(trialtype,bg_trialtype,dict_index_trialtype,dict_framenum_trialtype,dict_frameswitch_trialtype,
                                 denoised_individual_trials,) :
    index_trialtype = dict_index_trialtype[trialtype] 
    framenum_trialtype = dict_framenum_trialtype[trialtype] 
    frameswitch_trialtype = dict_frameswitch_trialtype[trialtype] 

    odor_on_frame,max_post_odor_frame,len_nan = find_align_position(framenum_trialtype,frameswitch_trialtype)
    num_go = len(framenum_trialtype)
    num_neurons = denoised_individual_trials['Trial1'].shape[0]
    

    order = np.argsort(index_trialtype)
    index_trialtype = np.sort(index_trialtype)
    framenum_trialtype = [ framenum_trialtype[i] for i in order]
    frameswitch_trialtype = [ frameswitch_trialtype[i] for i in order]

    
    neuron_baseline_std = std_in_baseline(num_neurons,bg_trialtype,denoised_individual_trials,dict_index_trialtype)
    F_dff_matrix1 = create_3d_matrix(num_neurons, len_nan, num_go, odor_on_frame, denoised_individual_trials, index_trialtype, framenum_trialtype,frameswitch_trialtype,neuron_baseline_std)
    return F_dff_matrix1,odor_on_frame,neuron_baseline_std

#%% import data
# load cnmf data
mouse_id  ='OT-GC-1'
date = '0518'
filedir = 'D:/PhD/Microscope/Selina/imaging_data/result/{}'.format(mouse_id)
sort_filenames1 = read_filenames_in_folder(filedir,'.pickle')
sort_filenames2 = read_filenames_in_folder(filedir,'.hdf5')
Dicts = load_files_into_dict(sort_filenames1)
CNMF_dict = load_files_into_dict(sort_filenames2)

if mouse_id == 'OT-GC-1':
    sessions = [1,2,3,4,5,6,8,9,10]
    trialtype = ['go']*5+['OdorReward']*4
elif mouse_id == 'OT-GC-3':
    sessions = [1,2,3,4,5,6,8,10] # for OT-GC-3
    trialtype = ['go']*5+['OdorReward']*3 # for OT-GC-3
    
    
big_goodor_list = []
big_water_list = []
big_whole_list = []
big_unpwater_list = []
spatial = []
for index in range(len(sessions)):
    
    cnmf = CNMF_dict[index]
    
    spatial_footprint = cnmf['estimates']['A']
    denoised_dff = detrend_df_f(cnmf['estimates']['A'], cnmf['estimates']['b'], cnmf['estimates']['C'], cnmf['estimates']['f'],
                                 cnmf['estimates']['YrA']*0.08, quantileMin=8, frames_window=1000, flag_auto=True, 
                                 use_fast=False, detrend_only=False)


    # load pickled data
    
    big_dict = Dicts[index]

    df_original = big_dict['behavior_original_df']
    df_all_drop = big_dict['behavior_dropped_df']
    
    # don't use this for D1
    #denoised_individual_trials = big_dict['temp_traces_trial_dict']
    dict_index_trialtype = big_dict['dict_index_trialtype']
    dict_framenum_trialtype = big_dict['dict_framenum_trialtype']
    #spatial_reg = big_dict['spatial_reg']
    dict_frameswitch_trialtype = big_dict['dict_frameswitch_trialtype']
    
    print(getKeys(dict_index_trialtype))

    denoised_individual_trials = slice_signal_to_trial(denoised_dff,df_original)   
    spatial.append(spatial_footprint)
    if index <=4:
        
        F_dff_matrix1,odor_on_frame1,neuron_baseline_std = genereate_dff_aligned_matrix('go','background',dict_index_trialtype,
                                                                                        dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                        denoised_individual_trials)
        F_dff_matrix2,odor_on_frame2,_ = genereate_dff_aligned_matrix('go_omit','background',dict_index_trialtype,
                                                                                        dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                        denoised_individual_trials,)
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+15)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+15)]),axis = 2)
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+15)]
        mat_go_whole = F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+30)]
        big_goodor_list.append(mat_go)
        big_water_list.append(mat_water)
        big_whole_list.append(mat_go_whole)
        
    else:
        F_dff_matrix1,odor_on_frame1,neuron_baseline_std = genereate_dff_aligned_matrix('OdorReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials)
        F_dff_matrix2,odor_on_frame2,_ = genereate_dff_aligned_matrix('OdorNReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
        F_dff_matrix3,odor_on_frame3,_ = genereate_dff_aligned_matrix('NOdorReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
        
        # percentage of responsive neurons
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+15)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+15)]),axis = 2)
        mat_go_whole = F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+30)]
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+15)]
        
        mat_unpwater = F_dff_matrix3[:,int(odor_on_frame3-6):int(odor_on_frame3+14)]
        big_goodor_list.append(mat_go)
        big_water_list.append(mat_water)
        big_whole_list.append(mat_go_whole)
        big_unpwater_list.append(mat_unpwater)
    #%%
templates_name_1 = ['D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0001_03-10-14-07_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0002_03-11-12-19_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0003_03-12-16-00_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0004_03-12-22-59_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0005_03-13-11-59_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0006_03-14-18-49_deg/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0008_03-15-23-44_deg/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0009_03-16-14-00_deg/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/00010_03-17-12-58_deg/AVG.png']
templates_name_3 = ['D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0001_03-11-15-17_cond/AVG.png',
                    'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0002_03-12-17-37_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0003_03-12-21-38_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0004_03-13-13-29_cond/AVG.png',
                    'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0005_03-13-22-56_cond/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0006_03-14-17-18_deg/AVG.png',
                    'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0008_03-15-22-48_deg/AVG.png',
                 'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/00011_03-17-14-04_deg/AVG.png']
# normalization
templates = []
for im_name in templates_name_1:
    temp = io.imread(im_name)
    vmax = temp.max()
    temp = temp/vmax
    templates.append(temp)
# dimension of the images
dims = templates[0].shape
#%%
spatial_union, assignments, matchings = register_multisession(A=spatial, dims=dims, templates=templates)
order = np.argsort(np.sum(np.isnan(assignments),axis = 1))
assignments_new = assignments[order,:]
assignments = assignments_new.copy()
assignments
#%%
n_reg = 8  # minimal number of sessions that each component has to be registered in

# Use number of non-NaNs in each row to filter out components that were not registered in enough sessions 
# (selina: confirmed, a row is a specific component in assignment matrix. and matching can tell you the kth component in J session is the ith component.)
assignments_filtered = np.array(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg], dtype=int);

from matplotlib.cm import get_cmap
alignments =  assignments_filtered  
# list_to_use = big_goodor_list
#list_to_use= big_water_list
list_to_use= big_whole_list

col = 8
row = int(np.ceil(alignments.shape[0]/col))
   
fig, axes = plt.subplots(row, col,sharex = True,figsize=(16,2*row))
ax = fig.add_axes([0.91, 0.15, .1, 0.7])   
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 

for i in range(alignments.shape[0]):    
    row_plot = int(np.floor(i/col))
    col_plot = i%col    
    # odor shaded area
    axes[row_plot, col_plot].fill_between([1,2], [10,10],[-10,-10],facecolor="grey", color='grey', alpha=0.1)   
    # lick shaded area
    axes[row_plot, col_plot].fill_between([4.5,4.58], [10,10],[-10,-10],facecolor="#3083D1", color='#3083D1',alpha=0.2)
    for j in range(alignments.shape[1]):
        mu = np.nanmean(list_to_use[j],axis=2)
        x = np.linspace(0,(mu.shape[1]-1)/5,mu.shape[1])
        try:
            color = cmap(j/alignments.shape[1])
            axes[row_plot, col_plot].plot(x,mu[int(alignments[i,j]),:], lw=2,  color = color, label = 'session {}'.format(sessions[j]),alpha=0.8)
        except:
            pass   
    axes[row_plot, col_plot].set_ylim(-2,4)
    handles, labels = axes[row_plot, col_plot].get_legend_handles_labels()
    if col_plot+row_plot == 0:
        ax.legend(handles, labels,loc='center left',frameon=False,fontsize = 18)
    axes[row_plot, col_plot].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False) 
    axes[row_plot, col_plot].spines["top"].set_visible(False)    
    axes[row_plot, col_plot].spines["right"].set_visible(False)
    axes[row_plot, col_plot].spines["bottom"].set_visible(False if row_plot != row-1 else True)    
    axes[row_plot, col_plot].spines["left"].set_visible(False if col_plot != 0 else True)    
    axes[row_plot, col_plot].set_xlabel('Time (s)' if row_plot == row-1 else '')
    axes[row_plot, col_plot].set_ylabel('Z-score' if col_plot == 0 else '')
ax.tick_params(axis="both", which="both", labelleft=False,labelbottom=False,bottom=False, left = False, top = False, right = False)
ax.spines["top"].set_visible(False)    
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)    
ax.spines["left"].set_visible(False)
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/over89session_gotrialwhole'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 200,transparent = True)
plt.show()
#%%
list_to_use= big_water_list
# list_to_use= big_whole_list

col = 8
row = int(np.ceil(alignments.shape[0]/col))
   
fig, axes = plt.subplots(row, col,sharex = True,figsize=(16,2*row))
ax = fig.add_axes([0.91, 0.15, .1, 0.7])   
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 

for i in range(alignments.shape[0]):    
    row_plot = int(np.floor(i/col))
    col_plot = i%col    
    # odor shaded area
    # axes[row_plot, col_plot].fill_between([1,2], [10,10],[-10,-10],facecolor="grey", color='grey', alpha=0.1)   
    # lick shaded area
    axes[row_plot, col_plot].fill_between([1,1.08], [10,10],[-10,-10],facecolor="#3083D1", color='#3083D1',alpha=0.2)
    for j in range(alignments.shape[1]):
        mu = np.nanmean(list_to_use[j],axis=2)
        mu = (mu.T - np.mean(mu[:,:5],axis=1)).T
        x = np.linspace(0,(mu.shape[1]-1)/5,mu.shape[1])
        try:
            color = cmap(j/alignments.shape[1])
            axes[row_plot, col_plot].plot(x,mu[int(alignments[i,j]),:], lw=2,  color = color, label = 'session {}'.format(sessions[j]),alpha=0.8)
        except:
            pass   
    axes[row_plot, col_plot].set_ylim(-3,2)
    handles, labels = axes[row_plot, col_plot].get_legend_handles_labels()
    if col_plot+row_plot == 0:
        ax.legend(handles, labels,loc='center left',frameon=False,fontsize = 18)
    axes[row_plot, col_plot].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False) 
    axes[row_plot, col_plot].spines["top"].set_visible(False)    
    axes[row_plot, col_plot].spines["right"].set_visible(False)
    axes[row_plot, col_plot].spines["bottom"].set_visible(False if row_plot != row-1 else True)    
    axes[row_plot, col_plot].spines["left"].set_visible(False if col_plot != 0 else True)    
    axes[row_plot, col_plot].set_xlabel('Time (s)' if row_plot == row-1 else '')
    axes[row_plot, col_plot].set_ylabel('Z-score' if col_plot == 0 else '')
ax.tick_params(axis="both", which="both", labelleft=False,labelbottom=False,bottom=False, left = False, top = False, right = False)
ax.spines["top"].set_visible(False)    
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)    
ax.spines["left"].set_visible(False)
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/over89session_water'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 200,transparent = True)
plt.show()
#%%
list_to_use= big_unpwater_list
# list_to_use= big_whole_list

col = 8
row = int(np.ceil(alignments.shape[0]/col))
   
fig, axes = plt.subplots(row, col,sharex = True,figsize=(16,2*row))
ax = fig.add_axes([0.91, 0.15, .1, 0.7])   
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 

for i in range(alignments.shape[0]):    
    row_plot = int(np.floor(i/col))
    col_plot = i%col    
    # odor shaded area
    # axes[row_plot, col_plot].fill_between([1,2], [10,10],[-10,-10],facecolor="grey", color='grey', alpha=0.1)   
    # lick shaded area
    axes[row_plot, col_plot].fill_between([1,1.08], [10,10],[-10,-10],facecolor="#3083D1", color='#3083D1',alpha=0.2)
    for j in range(alignments.shape[1]-5):
        mu = np.nanmean(list_to_use[j],axis=2)
        mu = (mu.T - np.mean(mu[:,:5],axis=1)).T
        x = np.linspace(0,(mu.shape[1]-1)/5,mu.shape[1])
        try:
            color = cmap((j+5)/alignments.shape[1])
            axes[row_plot, col_plot].plot(x,mu[int(alignments[i,j+5]),:], lw=2,  color = color, label = 'session {}'.format(sessions[j+5]),alpha=0.9)
        except:
            pass   
    axes[row_plot, col_plot].set_ylim(-1,2.5)
    handles, labels = axes[row_plot, col_plot].get_legend_handles_labels()
    if col_plot+row_plot == 0:
        ax.legend(handles, labels,loc='center left',frameon=False,fontsize = 18)
    axes[row_plot, col_plot].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False) 
    axes[row_plot, col_plot].spines["top"].set_visible(False)    
    axes[row_plot, col_plot].spines["right"].set_visible(False)
    axes[row_plot, col_plot].spines["bottom"].set_visible(False if row_plot != row-1 else True)    
    axes[row_plot, col_plot].spines["left"].set_visible(False if col_plot != 0 else True)    
    axes[row_plot, col_plot].set_xlabel('Time (s)' if row_plot == row-1 else '')
    axes[row_plot, col_plot].set_ylabel('Z-score' if col_plot == 0 else '')
ax.tick_params(axis="both", which="both", labelleft=False,labelbottom=False,bottom=False, left = False, top = False, right = False)
ax.spines["top"].set_visible(False)    
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)    
ax.spines["left"].set_visible(False)
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/over89session_unpredictedwater'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 200,transparent = True)
plt.show()
        