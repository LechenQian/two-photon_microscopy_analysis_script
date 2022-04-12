# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:26:04 2020

@author: qianl
"""
#%% import

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
        a[:,start_p:end_p,i] = sub*100
        
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

#%% template
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
#%% import data
# load cnmf data
pathlist_1 = ['D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0001_03-10-14-07_cond.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0002_03-11-12-19_cond.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0003_03-12-16-00_cond.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0004_03-12-22-59_cond.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0005_03-13-11-59_cond.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0006_03-14-18-49_deg.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0008_03-15-23-44_deg.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/0009_03-16-14-00_deg.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-1/00010_03-17-12-58_deg.hdf5']

pathlist_3 = ['D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0001_03-11-15-17_cond.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0002_03-12-17-37_cond.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0003_03-12-21-38_cond.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0004_03-13-13-29_cond.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0005_03-13-22-56_cond.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0006_03-14-17-18_deg.hdf5',
              'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/0008_03-15-22-48_deg.hdf5',
             'D:/PhD/Microscope/Selina/imaging_data/result/OT-GC-3/00011_03-17-14-04_deg.hdf5']


sessions = [1,2,3,4,5,6,8,9,10]
for index in range(len(sessions)):
    cnmfpath = pathlist_1[index]
    print(cnmfpath)
    cnmf = load_dict_from_hdf5(cnmfpath)
    spatial_footprint = cnmf['estimates']['A']
    denoised_dff = detrend_df_f(cnmf['estimates']['A'], cnmf['estimates']['b'], cnmf['estimates']['C'], cnmf['estimates']['f'],
                                 cnmf['estimates']['YrA']*0.08, quantileMin=8, frames_window=1000, flag_auto=True, 
                                 use_fast=False, detrend_only=False)


    # load pickled data
    picklepath = cnmfpath[:-5]+'.pickle'
    big_dict = load_pickleddata(picklepath)

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
    
    if index <=4:
        F_dff_matrix1,odor_on_frame1,neuron_baseline_std = genereate_dff_aligned_matrix('go','background',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials)
        F_dff_matrix2,odor_on_frame2,_ = genereate_dff_aligned_matrix('go_omit','background',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
        F_dff_matrix3,odor_on_frame3,_ = genereate_dff_aligned_matrix('no_go','background',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/OT-GC-1-new/000{}_example_mean'.format(sessions[index])
        display_multineuron_mean_std(F_dff_matrix1,odor_on_frame1,'go',
                                     F_dff_matrix2,odor_on_frame2,'go_omit',
                                     F_dff_matrix3,odor_on_frame3,'no_go',
                                      name = savename)
#         plt.figure(figsize = (10,10))
#         visualization.plot_contours(cnmf['estimates']['A'], templates[index])
#         plt.savefig('{}_contours.png'.format(savename))
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
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/OT-GC-1-new/000{}_example_mean'.format(sessions[index])
        display_multineuron_mean_std(F_dff_matrix1,odor_on_frame1,'OdorReward',
                                     F_dff_matrix2,odor_on_frame2,'OdorNReward',
                                     F_dff_matrix3,odor_on_frame3,'NOdorReward',
                                      name = savename)
#         plt.figure(figsize = (10,10))
#         visualization.plot_contours(cnmf['estimates']['A'], templates[index])
#         plt.savefig('{}_contours.png'.format(savename),bbox_inches="tight", dpi = 400,transparent = True)

#%% licking alignment (eventplot and line plot)
# load behavioral pickle data
def read_filenames_in_folder(filedir,postfix):
    
    filenames = []
    file_index = []
    for dirpath, dirnames, files in os.walk(filedir): # can walk through all levels down
        #     print(f'Found directory: {dirpath}')
        for f_name in files:
            if f_name.endswith('.pickle'):
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
        big_dict = load_pickleddata(filename)
        Dicts.append(big_dict)
        print(filename)
    return Dicts

mouse_id  ='OT-GC-3'
date = '0518'
filedir = 'D:/PhD/Microscope/Selina/imaging_data/result/{}'.format(mouse_id)
sort_filenames = read_filenames_in_folder(filedir,'.pickle')
Dicts = load_files_into_dict(sort_filenames)

# sessions = [1,2,3,4,5,6,8,10] # for OT-GC-3
sessions = [1,2,3,4,5,6,8,9,10] # for OT-GC-1

# trialtype = ['go']*5+['OdorReward']*3 # for OT-GC-3
trialtype = ['go']*5+['OdorReward']*4 # for OT-GC-1

# trialtype = ['go_omit']*5+['OdorNReward']*3
for index,big_dict in enumerate(Dicts):
    df_all_drop = big_dict['behavior_dropped_df'] ## adding df_all_drop
    index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]
    licking_time = df_all_drop.loc[index_trialtype,'licking'].tolist()
    odor_on_time = df_all_drop.loc[index_trialtype,'go_odor_on'].tolist()
    # realignment to 1s as odor on time
    new_licking_list = []
    lump_licking = []
    for i in range(len(index_trialtype)):
        try:
            shifted_licking_time = np.array(licking_time[i]) - odor_on_time[i][0]+1
        except:
            shifted_licking_time = np.array(licking_time[i]) - 2.5+1
        shifted_drop_licking = [x for x in shifted_licking_time if x <=7 and x >0 ]
        new_licking_list.append(shifted_drop_licking)
        lump_licking += shifted_drop_licking
        
    licking_df = {'shifted_licking':new_licking_list}
    dfObj = pd.DataFrame(licking_df) 
    bin_edges = np.linspace(0, 7, 16, endpoint=True)
    test,_ = np.histogram(lump_licking, bins=bin_edges)
    fig,ax = plt.subplots(figsize = (4,1))
    plt.plot(bin_edges[:-1]+0.25,test/len(index_trialtype)*2,COLOR = 'purple')
    plt.axvline(x=1,color = 'k')
    plt.axvline(x=2,color = 'k')
    plt.axvline(x=4.5,color = 'b')
    plt.ylim([0,4])
    plt.ylabel('licking rate(/s)')
    plt.xlabel('time(s)')
    plt.title('Session_{}_{}'.format(sessions[index],trialtype[index]),pad = 15)
              
    plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    
    savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_licking_go_hist'.format(mouse_id,date,sessions[index])
    plt.savefig(savename, bbox_inches="tight", dpi = 400)
    # plot
    fig,ax = plt.subplots(figsize = (4,2))
    plt.eventplot(dfObj.shifted_licking,color = 'grey')
    plt.axvline(x=1,color = 'k')
    plt.axvline(x=2,color = 'k')
    plt.axvline(x=4.5,color = 'b')
    plt.title('Session_{}_{}'.format(sessions[index],trialtype[index]),pad = 15)
    
    plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    
    plt.xlim([-0.5,7])
    plt.ylim([-0.8,len(index_trialtype)])
    plt.gca().invert_yaxis()
    
    
    savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_licking_go'.format(mouse_id,date,sessions[index])
    plt.savefig(savename, bbox_inches="tight", dpi = 400)
    plt.show()

#%% licking change on the same plot
mouse_id  ='OT-GC-3'
date = '0518'
filedir = 'D:/PhD/Microscope/Selina/imaging_data/result/{}'.format(mouse_id)
sort_filenames = read_filenames_in_folder(filedir,'.pickle')
Dicts = load_files_into_dict(sort_filenames)

if mouse_id == 'OT-GC-1':
    sessions = [1,2,3,4,5,6,8,9,10]
    trialtype = ['go']*5+['OdorReward']*4
elif mouse_id == 'OT-GC-3':
    sessions = [1,2,3,4,5,6,8,10] # for OT-GC-3
    trialtype = ['go']*5+['OdorReward']*3 # for OT-GC-3

fig,ax = plt.subplots(figsize = (6,4))
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 
# trialtype = ['go_omit']*5+['OdorNReward']*3
for index,big_dict in enumerate(Dicts):
    if index in [0,3,5,7]:
        df_all_drop = big_dict['behavior_dropped_df'] ## adding df_all_drop
        index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]
        licking_time = df_all_drop.loc[index_trialtype,'licking'].tolist()
        odor_on_time = df_all_drop.loc[index_trialtype,'go_odor_on'].tolist()
        # realignment to 1s as odor on time
        new_licking_list = []
        lump_licking = []
        for i in range(len(index_trialtype)):
            try:
                shifted_licking_time = np.array(licking_time[i]) - odor_on_time[i][0]+1
            except:
                shifted_licking_time = np.array(licking_time[i]) - 2.5+1
            shifted_drop_licking = [x for x in shifted_licking_time if x <=7 and x >0 ]
            
            lump_licking += shifted_drop_licking
            
        
        bin_edges = np.linspace(0, 7, 16, endpoint=True)
        test,_ = np.histogram(lump_licking, bins=bin_edges)
        
        color = cmap(index/len(Dicts))
        plt.plot(bin_edges[:-1]+0.25,test/len(index_trialtype)*2,color = color,label = 'session {}'.format(sessions[index]))
        plt.axvline(x=1,color = 'k')
        plt.axvline(x=2,color = 'k')
        plt.axvline(x=4.5,color = 'b')
        plt.ylim([0,4])
        plt.xlim([0,8])
        plt.ylabel('licking rate(/s)')
        plt.xlabel('time(s)')
        plt.title('{}_all_session'.format(mouse_id),pad = 15)
        plt.legend(loc = 1,fontsize = 10)
                  
        plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)
    
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_licking_go_hist_13810'.format(mouse_id,date,sessions[index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400)

#%% background
mouse_id  ='OT-GC-1'
date = '0518'
filedir = 'D:/PhD/Microscope/Selina/imaging_data/result/{}'.format(mouse_id)
sort_filenames = read_filenames_in_folder(filedir,'.pickle')
Dicts = load_files_into_dict(sort_filenames)

if mouse_id == 'OT-GC-1':
    sessions = [1,2,3,4,5,6,8,9,10]
    trialtype = ['background']*5+['NOdorNReward']*4
elif mouse_id == 'OT-GC-3':
    sessions = [1,2,3,4,5,6,8,10] # for OT-GC-3
    trialtype = ['background']*5+['NOdorNReward']*3 # for OT-GC-3



for index,big_dict in enumerate(Dicts):
    df_all_drop = big_dict['behavior_dropped_df'] ## adding df_all_drop
    if trialtype[index] == 'background':
        index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]
    else:
        index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]+ big_dict['dict_index_trialtype']['NOdorReward']
        index_trialtype.sort()
    licking_time = df_all_drop.loc[index_trialtype,'licking'].tolist()
    
    # realignment to 1s as odor on time
    new_licking_list = []
    lump_licking = []
    for i in range(len(index_trialtype)):
        shifted_licking_time = np.array(licking_time[i]) #- 2.5+1
        shifted_drop_licking = [x for x in shifted_licking_time if x <=6 and x >0 ]
        new_licking_list.append(shifted_drop_licking)
        lump_licking += shifted_drop_licking
        
    licking_df = {'shifted_licking':new_licking_list}
    dfObj = pd.DataFrame(licking_df) 
    bin_edges = np.linspace(0, 6, 14, endpoint=True)
    test,_ = np.histogram(lump_licking, bins=bin_edges)
    fig,ax = plt.subplots(figsize = (4,1))
    
    plt.plot(bin_edges[:-1]+0.25,test/len(index_trialtype)*2,color = 'purple')
    # if trialtype[index] != 'background':
    
    #     plt.axvline(x=6.1,color = 'b')
    plt.ylim([0,0.8])
    plt.xlim([0,10])
    plt.ylabel('licking rate(/s)')
    plt.xlabel('time(s)')
    plt.title('Session_{}_{}'.format(sessions[index],trialtype[index]),pad = 15)
              
    plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    
    savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_licking_background_hist_beforewater'.format(mouse_id,date,sessions[index])
    plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)

    # # plot
    # fig,ax = plt.subplots(figsize = (4,2))
    # plt.eventplot(dfObj.shifted_licking,color = 'grey')
    # #plt.axvline(x=2.6,color = 'k')
    # #plt.axvline(x=3.6,color = 'k')
    # if trialtype[index] != 'background':
    #     plt.axvline(x=6.1,color = 'b')
    # plt.title('Session_{}_{}'.format(sessions[index],trialtype[index]),pad = 15)
    
    # plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
    # ax.spines["top"].set_visible(False)    
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)    
    # ax.spines["left"].set_visible(False)
    
    # plt.xlim([-0.5,10])
    # plt.ylim([-0.8,len(index_trialtype)])
    # plt.gca().invert_yaxis()
    
    
    # savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/OT-GC-1-new/000{}_licking_background'.format(sessions[index])

    # plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
    # plt.show()




#%%
mouse_id  ='OT-GC-1'
date = '0518'
filedir = 'D:/PhD/Microscope/Selina/imaging_data/result/{}'.format(mouse_id)
sort_filenames = read_filenames_in_folder(filedir,'.pickle')
Dicts = load_files_into_dict(sort_filenames)

if mouse_id == 'OT-GC-1':
    sessions = [1,2,3,4,5,6,8,9,10]
    trialtype = ['background']*5+['NOdorNReward']*4
elif mouse_id == 'OT-GC-3':
    sessions = [1,2,3,4,5,6,8,10] # for OT-GC-3
    trialtype = ['background']*5+['NOdorNReward']*3 # for OT-GC-3


fig,ax = plt.subplots(figsize = (8,4))
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 
for index,big_dict in enumerate(Dicts):
    df_all_drop = big_dict['behavior_dropped_df'] ## adding df_all_drop
    if trialtype[index] == 'background':
        index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]
    else:
        index_trialtype = big_dict['dict_index_trialtype'][trialtype[index]]+ big_dict['dict_index_trialtype']['NOdorReward']
        index_trialtype.sort()
    licking_time = df_all_drop.loc[index_trialtype,'licking'].tolist()
    
    # realignment to 1s as odor on time
    new_licking_list = []
    lump_licking = []
    for i in range(len(index_trialtype)):
        shifted_licking_time = np.array(licking_time[i]) #- 2.5+1
        shifted_drop_licking = [x for x in shifted_licking_time if x <=6 and x >0 ]
        
        lump_licking += shifted_drop_licking
        
    
    bin_edges = np.linspace(0, 6, 7, endpoint=True)
    test,_ = np.histogram(lump_licking, bins=bin_edges)
    
    color = cmap(index/len(Dicts))
    plt.plot(bin_edges[:-1]+0.25,test/len(index_trialtype),color= color,label = 'session {}'.format(sessions[index]))
    
    plt.ylim([0,0.6])
    plt.xlim([0,7])
    plt.ylabel('licking rate(/s)')
    plt.xlabel('time(s)')
    plt.title('{}_background_all'.format(mouse_id),pad = 15)

    plt.legend(loc = 1,fontsize = 10)
     
    plt.tick_params(axis="both", which="both", labelleft=True ,bottom=False, left = False, top = False, right = False) 
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_licking_background_hist_all'.format(mouse_id,date,sessions[index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
































