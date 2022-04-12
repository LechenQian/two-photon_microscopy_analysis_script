# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:01:53 2020

@author: qianl
"""


#%% import
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

#%% template
# templates_name_1 = ['D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0001_03-10-14-07_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0002_03-11-12-19_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0003_03-12-16-00_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0004_03-12-22-59_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0005_03-13-11-59_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0006_03-14-18-49_deg/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0008_03-15-23-44_deg/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0009_03-16-14-00_deg/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/00010_03-17-12-58_deg/AVG.png']
# templates_name_3 = ['D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0001_03-11-15-17_cond/AVG.png',
#                     'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0002_03-12-17-37_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0003_03-12-21-38_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0004_03-13-13-29_cond/AVG.png',
#                     'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0005_03-13-22-56_cond/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0006_03-14-17-18_deg/AVG.png',
#                     'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/0008_03-15-22-48_deg/AVG.png',
#                  'D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-3/00011_03-17-14-04_deg/AVG.png']


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
mouse_id  ='OT-GC-1'
date = '0527'
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

        
        # percentage of responsive neurons
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+10)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+10)]),axis = 2)
        mat_nogo = F_dff_matrix3[:,int(odor_on_frame3-5):int(odor_on_frame3+10)]
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+10)]
        
        
        mean_nogo = np.nanmean(mat_nogo,axis = 2)
        mean_nogo = (mean_nogo.T - np.mean(mean_nogo[:,:5],axis=1)).T
        mean_water = np.nanmean(mat_water,axis = 2)
        mean_water = (mean_water.T - np.mean(mean_water[:,:5],axis=1)).T
        mean_go = np.nanmean(mat_go,axis = 2)
        mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
        
        
        mean_go_compreg = mean_go[:,5:]
        mean_nogo_compreg = mean_nogo[:,5:]
        mean_water_compreg = mean_water[:,5:]
        # maximual value of absolute response
        max_go = np.max(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))
        max_nogo = np.max(np.abs(mean_nogo_compreg),axis = 1).reshape((1,-1))
        max_water = np.max(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))
        #
        comp_mat = np.concatenate((max_go, max_nogo,max_water))
        ind = np.argmax(comp_mat,axis = 0)
        group_go = [x for x,g in enumerate(ind) if g==0]
        group_nogo = [x for x,g in enumerate(ind) if g==1]
        group_water = [x for x,g in enumerate(ind) if g==2]
        #
        order_go = np.argsort(-np.mean(mean_go_compreg[group_go,:],axis = 1))
        order_nogo = np.argsort(-np.mean(mean_nogo_compreg[group_nogo,:],axis = 1))
        order_water = np.argsort(-np.mean(mean_water_compreg[group_water,:],axis = 1))
        
        #
        whole_mat = np.hstack((mean_go,mean_water,mean_nogo,))
        #
        sel_go = whole_mat[group_go,:]
        sel_nogo = whole_mat[group_nogo,:]
        sel_water = whole_mat[group_water,:]
        sel_go = sel_go[order_go,:]
        sel_nogo = sel_nogo[order_nogo,:]
        sel_water = sel_water[order_water,:]
        sorted_whole_mat = np.vstack((sel_go,sel_water,sel_nogo))
        #
        
        
        cax = sns.heatmap(sorted_whole_mat,cmap = 'RdBu_r',vmin = -2,vmax = 2)
        kwargs = {'linestyle':'--','linewidth':2,'color':'grey'}
        plt.axvline(x=5,**kwargs),plt.axvline(x=15,**kwargs)
        plt.axvline(x=20,**kwargs),plt.axvline(x=30,**kwargs)
        plt.axvline(x=35,**kwargs),plt.axvline(x=45,**kwargs)
        plt.axhline(y=len(group_go),**kwargs),plt.axhline(y=len(group_go)+len(group_water),**kwargs)
        plt.yticks(np.arange(0,sorted_whole_mat.shape[0],10),np.arange(0,sorted_whole_mat.shape[0],10))
        plt.xticks([],[])
        plt.title('session {}'.format(sessions[index]),pad  =20)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=10)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_go_nogo_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
        plt.show()
    

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
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+10)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+10)]),axis = 2)
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+10)]
        mat_unpwater = F_dff_matrix3[:,int(odor_on_frame3-6):int(odor_on_frame3+9)]
        
        mean_go = np.nanmean(mat_go,axis = 2)
        mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
        mean_unpwater = np.nanmean(mat_unpwater,axis = 2)
        mean_unpwater = (mean_unpwater.T - np.mean(mean_unpwater[:,:5],axis=1)).T
        mean_water = np.nanmean(mat_water,axis = 2)
        mean_water = (mean_water.T - np.mean(mean_water[:,:5],axis=1)).T
        
        mean_go_compreg = mean_go[:,5:]
        mean_unpwater_compreg = mean_unpwater[:,5:]
        mean_water_compreg = mean_water[:,5:]
        # maximual value of absolute response
        max_go = np.max(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))
        max_unpwater = np.max(np.abs(mean_unpwater_compreg),axis = 1).reshape((1,-1))
        max_water = np.max(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))
        #
        comp_mat = np.concatenate((max_go, max_unpwater,max_water))
        ind = np.argmax(comp_mat,axis = 0)
        group_go = [x for x,g in enumerate(ind) if g==0]
        group_unpwater = [x for x,g in enumerate(ind) if g==1]
        group_water = [x for x,g in enumerate(ind) if g==2]
        #
        order_go = np.argsort(-np.mean(mean_go_compreg[group_go,:],axis = 1))
        order_unpwater = np.argsort(-np.mean(mean_unpwater_compreg[group_unpwater,:],axis = 1))
        order_water = np.argsort(-np.mean(mean_water_compreg[group_water,:],axis = 1))
        
        #
        whole_mat = np.hstack((mean_go,mean_water,mean_unpwater))
        #
        sel_go = whole_mat[group_go,:]
        sel_unpwater = whole_mat[group_unpwater,:]
        sel_water = whole_mat[group_water,:]
        sel_go = sel_go[order_go,:]
        sel_unpwater = sel_unpwater[order_unpwater,:]
        sel_water = sel_water[order_water,:]
        sorted_whole_mat = np.vstack((sel_go,sel_water,sel_unpwater))
        #
        
        
        cax = sns.heatmap(sorted_whole_mat,cmap = 'RdBu_r',vmin = -2,vmax = 2)
        kwargs = {'linestyle':'--','linewidth':2,'color':'grey'}
        plt.axvline(x=5,**kwargs),plt.axvline(x=15,**kwargs)
        plt.axvline(x=20,**kwargs),plt.axvline(x=30,**kwargs)
        plt.axvline(x=35,**kwargs),plt.axvline(x=45,**kwargs)
        
        plt.axhline(y=len(group_go),**kwargs),plt.axhline(y=len(group_go)+len(group_water),**kwargs)
        plt.yticks(np.arange(0,sorted_whole_mat.shape[0],10),np.arange(0,sorted_whole_mat.shape[0],10))
        plt.xticks([],[])
        plt.title('session {}'.format(sessions[index]),pad  =20)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=10)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_go_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
        plt.show()
    
#%% calculate percentage
# load cnmf data
mouse_id  ='OT-GC-3'
date = '0519'
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

        
        # percentage of responsive neurons
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+10)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+10)]),axis = 2)
        mat_nogo = F_dff_matrix3[:,int(odor_on_frame3-5):int(odor_on_frame3+10)]
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+10)]
        
        
        mean_nogo = np.nanmean(mat_nogo,axis = 2)
        mean_nogo = (mean_nogo.T - np.mean(mean_nogo[:,:5],axis=1)).T
        mean_water = np.nanmean(mat_water,axis = 2)
        mean_water = (mean_water.T - np.mean(mean_water[:,:5],axis=1)).T
        mean_go = np.nanmean(mat_go,axis = 2)
        mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
        
        
        mean_go_compreg = mean_go[:,5:]
        mean_nogo_compreg = mean_nogo[:,5:]
        mean_water_compreg = mean_water[:,5:]
        # maximual value of absolute response
        max_go_index = np.argmax(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))[0]
        max_nogo_index = np.argmax(np.abs(mean_nogo_compreg),axis = 1).reshape((1,-1))[0]
        max_water_index = np.argmax(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))[0]
        # 
#

        magnitude_go = np.array([mean_go_compreg[i,k] for i,k in enumerate(max_go_index)])
        magnitude_nogo = np.array([mean_nogo_compreg[i,k] for i,k in enumerate(max_nogo_index)])
        magnitude_water = np.array([mean_water_compreg[i,k] for i,k in enumerate(max_water_index)])
        
        bars1 = activate = [sum(magnitude_go>0.5)/len(magnitude_go),sum(magnitude_water>0.5)/len(magnitude_water),sum(magnitude_nogo>0.5)/len(magnitude_nogo)]
        bars2 = supress = [-sum(magnitude_go<-0.5)/len(magnitude_go),-sum(magnitude_water<-0.5)/len(magnitude_water),-sum(magnitude_nogo<-0.5)/len(magnitude_nogo)]
        
        fig,ax = plt.subplots(figsize=(3.8,3))
        
        # The position of the bars on the x-axis
        x = [1,2,3]
         
        # Names of group and bar width
        names = ['go odor','predited water','no-go odor']
        barWidth = 0.8
         
        # Create brown bars
        plt.bar(x, bars1, color='#FFD906', edgecolor='k', width=barWidth,label = 'activated')
        # Create green bars (middle), on top of the firs ones
        plt.bar(x, bars2, color='#D17DF5', edgecolor='k', width=barWidth,label = 'inhibited')
        
         
        # Custom X axis
        plt.xticks(x, names)#, fontweight='bold')
        plt.xlabel("Stimulus")
        plt.yticks([-0.4,-0.2,0,0.2,0.4], [-0.4,-0.2,0,0.2,0.4])
        plt.ylim([-0.5,0.5])
        plt.ylabel('Stimulus responsive')
        plt.title('session {}'.format(sessions[index]),pad  =20)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_percentage_go_nogo_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        # Show graphic
        plt.show()

        
        max_go = np.max(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))
        max_nogo = np.max(np.abs(mean_nogo_compreg),axis = 1).reshape((1,-1))
        max_water = np.max(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))
        comp_mat = np.concatenate((max_go, max_nogo,max_water))
        ind = np.argmax(comp_mat,axis = 0)
        group_go = [x for x,g in enumerate(ind) if g==0]
        group_nogo = [x for x,g in enumerate(ind) if g==1]
        group_water = [x for x,g in enumerate(ind) if g==2]
        
        #
        sel_go = magnitude_go[group_go]
        sel_nogo = magnitude_nogo[group_nogo]
        sel_water = magnitude_water[group_water]
        
        #
        bars1 = activate = [sum(sel_go>0.5)/len(sel_go),sum(sel_water>0.5)/len(sel_water),sum(sel_nogo>0.5)/len(sel_nogo)]
        bars2 = supress = [-sum(sel_go<-0.5)/len(sel_go),-sum(sel_water<-0.5)/len(sel_water),-sum(sel_nogo<-0.5)/len(sel_nogo)]
        
        fig,ax = plt.subplots(figsize=(3.8,3))
        
        # The position of the bars on the x-axis
        x = [1,2,3]
         
        # Names of group and bar width
        names = ['go odor','predited water','no-go odor']
        barWidth = 0.8
         
        # Create brown bars
        plt.bar(x, bars1, color='#FFD906', edgecolor='k', width=barWidth,label = 'activated')
        # Create green bars (middle), on top of the firs ones
        plt.bar(x, bars2, color='#D17DF5', edgecolor='k', width=barWidth,label = 'inhibited')
        
         
        # Custom X axis
        plt.xticks(x, names)#, fontweight='bold')
        plt.xlabel("Stimulus")
        plt.yticks([-0.4,-0.2,0,0.2,0.4], [-0.4,-0.2,0,0.2,0.4])
        plt.ylim([-0.5,0.5])
        plt.ylabel('prefered stimulus')
        plt.title('session {}'.format(sessions[index]),pad  =20)
        
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_prefered_go_nogo_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        # Show graphic
        plt.show()
#
        fig,ax = plt.subplots(figsize=(3.5,3.5))
        
        
        
        plt.axvline(x=0,linestyle='--',color ='k')       
        plt.axhline(y=0,linestyle='--',color ='k') 
        vmax = 3.5#np.max([magnitude_go,magnitude_nogo])
        vmin = -2.5#np.min([magnitude_go,magnitude_nogo])
        y=x = np.linspace(vmin-0.2,vmax+0.2,100) 
        plt.plot(x,y,'--',color ='k')
        
        plt.scatter(magnitude_go,magnitude_nogo,edgecolor = 'k',color ='red',s = 15)  
        plt.xlabel('go odor magnitude (Z-score)')
        
        plt.ylabel('no go odor magnitude (Z-score)')
        plt.xlim([-2.5,3.5])
        plt.ylim([-2.5,3.5])
        plt.title('session {}'.format(sessions[index]),pad  =20)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_scatter_go_nogo'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        plt.show()

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
        mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+10)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+10)]),axis = 2)
        mat_water = F_dff_matrix1[:,int(odor_on_frame1+17-5):int(odor_on_frame1+17+10)]
        mat_unpwater = F_dff_matrix3[:,int(odor_on_frame3-6):int(odor_on_frame3+9)]
        
        mean_go = np.nanmean(mat_go,axis = 2)
        mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
        mean_unpwater = np.nanmean(mat_unpwater,axis = 2)
        mean_unpwater = (mean_unpwater.T - np.mean(mean_unpwater[:,:5],axis=1)).T
        mean_water = np.nanmean(mat_water,axis = 2)
        mean_water = (mean_water.T - np.mean(mean_water[:,:5],axis=1)).T
        
        mean_go_compreg = mean_go[:,5:]
        mean_unpwater_compreg = mean_unpwater[:,5:]
        mean_water_compreg = mean_water[:,5:]
        # index of maximual value of absolute response
        max_go_index = np.argmax(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))[0]
        max_unpwater_index = np.argmax(np.abs(mean_unpwater_compreg),axis = 1).reshape((1,-1))[0]
        max_water_index = np.argmax(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))[0]
        
        magnitude_go = np.array([mean_go_compreg[i,k] for i,k in enumerate(max_go_index)])
        magnitude_unpwater = np.array([mean_unpwater_compreg[i,k] for i,k in enumerate(max_unpwater_index)])
        magnitude_water = np.array([mean_water_compreg[i,k] for i,k in enumerate(max_water_index)])
        
        bars1 = activate = [sum(magnitude_go>0.5)/len(magnitude_go),sum(magnitude_water>0.5)/len(magnitude_water),sum(magnitude_unpwater>0.5)/len(magnitude_unpwater)]
        bars2 = supress = [-sum(magnitude_go<-0.5)/len(magnitude_go),-sum(magnitude_water<-0.5)/len(magnitude_water),-sum(magnitude_unpwater<-0.5)/len(magnitude_unpwater)]
        
        fig,ax = plt.subplots(figsize=(3.8,3))
        
        # The position of the bars on the x-axis
        x = [1,2,3]
         
        # Names of group and bar width
        names = ['go odor','pred water','unpred water']
        barWidth = 0.8
         
        # Create brown bars
        plt.bar(x, bars1, color='#FFD906', edgecolor='k', width=barWidth,label = 'activated')
        # Create green bars (middle), on top of the firs ones
        plt.bar(x, bars2, color='#D17DF5', edgecolor='k', width=barWidth,label = 'inhibited')
        
         
        # Custom X axis
        plt.xticks(x, names)#, fontweight='bold')
        plt.xlabel("Stimulus")
        plt.yticks([-0.4,-0.2,0,0.2,0.4], [-0.4,-0.2,0,0.2,0.4])
        plt.ylim([-0.5,0.5])
        plt.ylabel('Stimulus responsive')
        plt.title('session {}'.format(sessions[index]),pad  =20)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_percentage_go_nogo_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        # Show graphic
        plt.show()
        
        
        
        
        #
        max_go = np.max(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))
        max_unpwater = np.max(np.abs(mean_unpwater_compreg),axis = 1).reshape((1,-1))
        max_water = np.max(np.abs(mean_water_compreg),axis = 1).reshape((1,-1))
        comp_mat = np.concatenate((max_go, max_unpwater,max_water))
        ind = np.argmax(comp_mat,axis = 0)
        group_go = [x for x,g in enumerate(ind) if g==0]
        group_unpwater = [x for x,g in enumerate(ind) if g==1]
        group_water = [x for x,g in enumerate(ind) if g==2]
        
        #
        sel_go = magnitude_go[group_go]
        sel_unpwater = magnitude_unpwater[group_unpwater]
        sel_water = magnitude_water[group_water]
        
        #
        bars1 = activate = [sum(sel_go>0.5)/len(sel_go),sum(sel_water>0.5)/len(sel_water),sum(sel_unpwater>0.5)/len(sel_unpwater)]
        bars2 = supress = [-sum(sel_go<-0.5)/len(sel_go),-sum(sel_water<-0.5)/len(sel_water),-sum(sel_unpwater<-0.5)/len(sel_unpwater)]
        
        fig,ax = plt.subplots(figsize=(3.8,3))
        
        # The position of the bars on the x-axis
        x = [1,2,3]
         
        # Names of group and bar width
        names = ['go odor','pred water','unpred water']
        barWidth = 0.8
         
        # Create brown bars
        plt.bar(x, bars1, color='#FFD906', edgecolor='k', width=barWidth,label = 'activated')
        # Create green bars (middle), on top of the firs ones
        plt.bar(x, bars2, color='#D17DF5', edgecolor='k', width=barWidth,label = 'inhibited')
        
         
        # Custom X axis
        plt.xticks(x, names)#, fontweight='bold')
        plt.xlabel("Stimulus")
        plt.yticks([-0.4,-0.2,0,0.2,0.4], [-0.4,-0.2,0,0.2,0.4])
        plt.ylim([-0.5,0.5])
        plt.ylabel('prefered stimulus')
        plt.title('session {}'.format(sessions[index]),pad  =20)
        
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_prefered_go_nogo_water'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        # Show graphic
        plt.show()
#
        fig,ax = plt.subplots(figsize=(3.5,3.5))
        
        
        vmax = 10#np.max([magnitude_water,magnitude_unpwater])
        vmin = -2#np.min([magnitude_water,magnitude_unpwater])
        y=x = np.linspace(vmin,vmax,100) 
        plt.axvline(x=0,linestyle='--',color ='k')       
        plt.axhline(y=0,linestyle='--',color ='k') 
        y=x = np.linspace(vmin-0.2,vmax+0.2,100) 
        plt.plot(x,y,'--',color ='k')
        
        plt.scatter(magnitude_water,magnitude_unpwater,edgecolor = 'k',color ='red',s = 15)  
        plt.xlabel('predicted water magnitude (Z-score)')
        
        plt.ylabel('unpredicted water magnitude (Z-score)')
        plt.xlim([-2,10])
        plt.ylim([-2,10])
        plt.title('session {}'.format(sessions[index]),pad  =20)
        savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/000{}_scatter_unpwater_predwater'.format(mouse_id,date,sessions[index])
        plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
        plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True) 
        plt.show()


#%% FIND SIGNAFICANT ACTIVATED OR INHIBITED NEURONS AND THEN CALCULATE THE MEAN OF THEIR ACTIVITIES
#I'm only looking at the the go odor, so don't need to worry to much about the conditioning and degradation
#the neuronsneedn't have to be registered.

# let's get started!

#%% part 1
mouse_id  ='OT-GC-1'
date = '0527'
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
signals = []
for index in range(len(sessions)):   
    cnmf = CNMF_dict[index]    
    spatial_footprint = cnmf['estimates']['A']
    denoised_dff = detrend_df_f(cnmf['estimates']['A'], cnmf['estimates']['b'], cnmf['estimates']['C'], cnmf['estimates']['f'],
                                 cnmf['estimates']['YrA']*0.08, quantileMin=8, frames_window=1000, flag_auto=True, 
                                 use_fast=False, detrend_only=False)
    signals.append(denoised_dff)
#%% part 2 all together plot
#create a plot
fig, axes = plt.subplots(1,1,figsize=(4,5))
for index in range(len(sessions)):   

    # load pickled data    
    big_dict = Dicts[index]
    df_original = big_dict['behavior_original_df']
    df_all_drop = big_dict['behavior_dropped_df']        
    dict_index_trialtype = big_dict['dict_index_trialtype']
    dict_framenum_trialtype = big_dict['dict_framenum_trialtype']
    dict_frameswitch_trialtype = big_dict['dict_frameswitch_trialtype']    
    print(getKeys(dict_index_trialtype))
    denoised_individual_trials = slice_signal_to_trial(signals[index],df_original)   
    
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
    else:
        F_dff_matrix1,odor_on_frame1,neuron_baseline_std = genereate_dff_aligned_matrix('OdorReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials)
        F_dff_matrix2,odor_on_frame2,_ = genereate_dff_aligned_matrix('OdorNReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
    
    ## mean activities of responsive neurons
    # extrat signals in response to go and no-go odor
    mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+15)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+15)]),axis = 2)
    mat_nogo = F_dff_matrix3[:,int(odor_on_frame3-5):int(odor_on_frame3+15)]
    
    # calculate mean across trials    
    mean_nogo = np.nanmean(mat_nogo,axis = 2)
    mean_nogo = (mean_nogo.T - np.mean(mean_nogo[:,:5],axis=1)).T
    mean_go = np.nanmean(mat_go,axis = 2)
    mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
    
    # region for comparison
    mean_go_compreg = mean_go[:,5:]
    mean_nogo_compreg = mean_nogo[:,5:]
        
    # index of maximual value of absolute response
    max_go_index = np.argmax(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))[0]
    max_nogo_index = np.argmax(np.abs(mean_nogo_compreg),axis = 1).reshape((1,-1))[0]
    
    magnitude_go = np.array([mean_go_compreg[i,k] for i,k in enumerate(max_go_index)])
    magnitude_nogo = np.array([mean_nogo_compreg[i,k] for i,k in enumerate(max_nogo_index)])
    
    # activated neurons
    mean_go_act = mean_go[magnitude_go>0.5,:]
    mean_nogo_act = mean_nogo[magnitude_nogo>0.5,:]
    # inhibited neurons
    mean_go_inh = mean_go[magnitude_go <-0.5,:]
    mean_nogo_inh = mean_nogo[magnitude_nogo <-0.5,:]
    #mean of subgroup
    mean_sub_go_act = np.nanmean(mean_go_act,axis = 0)
    std_sub_go_act = np.nanstd(mean_go_act,axis = 0)/mean_go_act.shape[1]
    mean_sub_nogo_act = np.nanmean(mean_nogo_act,axis = 0)
    std_sub_nogo_act = np.nanstd(mean_nogo_act,axis = 0)/mean_nogo_act.shape[1]
    mean_sub_go_inh = np.nanmean(mean_go_inh,axis = 0)
    std_sub_go_inh = np.nanstd(mean_go_inh,axis = 0)/mean_go_inh.shape[1]
    mean_sub_nogo_inh = np.nanmean(mean_nogo_inh,axis = 0)
    std_sub_nogo_inh = np.nanstd(mean_nogo_inh,axis = 0)/mean_nogo_inh.shape[1]

    # figure plot    
    
    ax = fig.add_axes([0.91, 0.15, .05, 0.7])   
    colormap = "coolwarm"
    cmap = mpl.cm.get_cmap(colormap) 
 
    x = np.linspace(0,(len(mean_sub_go_act)-1)/5,len(mean_sub_go_act))
    color = cmap(index/len(sessions))
    axes.plot(x,mean_sub_go_act, lw=2,  color = color, label = 'session {}'.format(sessions[index]),alpha=0.8)
    axes.fill_between(x,mean_sub_go_act+std_sub_go_act,mean_sub_go_act-std_sub_go_act, facecolor= color, alpha=0.4) 
    axes.plot(x,mean_sub_go_inh, lw=2,  color = color, alpha=0.8)
    axes.fill_between(x,mean_sub_go_inh+std_sub_go_inh,mean_sub_go_inh-std_sub_go_inh, facecolor= color, alpha=0.4) 
    
    axes.set_ylim(-1.2,1.5)   
    axes.tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False) 
    axes.spines["top"].set_visible(False)    
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False )    
    axes.spines["left"].set_visible(False )    
    axes.set_xlabel('Time (s)' )
    axes.set_ylabel('Z-score' )
axes.fill_between([1,2], [10,10],[-10,-10],facecolor="grey", color='grey',alpha=0.2)
axes.axhline(y=0,color = 'k',alpha = 0.7)
handles, labels = axes.get_legend_handles_labels()   
ax.legend(handles, labels,loc='center left',frameon=False,fontsize = 12)
ax.tick_params(axis="both", which="both", labelleft=False,labelbottom=False,bottom=False, left = False, top = False, right = False)
ax.spines["top"].set_visible(False)    
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)    
ax.spines["left"].set_visible(False)
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/mean_activities_activated_inhibited_neurons'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()
#%% part 3 separate plots
#create a plot
fig, axes = plt.subplots(1,len(sessions),figsize=(20,4))
for index in range(len(sessions)):   

    # load pickled data    
    big_dict = Dicts[index]
    df_original = big_dict['behavior_original_df']
    df_all_drop = big_dict['behavior_dropped_df']        
    dict_index_trialtype = big_dict['dict_index_trialtype']
    dict_framenum_trialtype = big_dict['dict_framenum_trialtype']
    dict_frameswitch_trialtype = big_dict['dict_frameswitch_trialtype']    
    print(getKeys(dict_index_trialtype))
    denoised_individual_trials = slice_signal_to_trial(signals[index],df_original)   
    
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
    else:
        F_dff_matrix1,odor_on_frame1,neuron_baseline_std = genereate_dff_aligned_matrix('OdorReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials)
        F_dff_matrix2,odor_on_frame2,_ = genereate_dff_aligned_matrix('OdorNReward','NOdorNReward',dict_index_trialtype,
                                                                                       dict_framenum_trialtype,dict_frameswitch_trialtype,
                                                                                       denoised_individual_trials,)
    
    ## mean activities of responsive neurons
    # extrat signals in response to go and no-go odor
    mat_go = np.concatenate((F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+17)],F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+17)]),axis = 2)
    mat_nogo = F_dff_matrix3[:,int(odor_on_frame3-5):int(odor_on_frame3+17)]
    
    # calculate mean across trials    
    mean_nogo = np.nanmean(mat_nogo,axis = 2)
    mean_nogo = (mean_nogo.T - np.mean(mean_nogo[:,:5],axis=1)).T
    mean_go = np.nanmean(mat_go,axis = 2)
    mean_go = (mean_go.T - np.mean(mean_go[:,:5],axis=1)).T
    
    # region for comparison
    mean_go_compreg = mean_go[:,5:]
    mean_nogo_compreg = mean_nogo[:,5:]
        
    # index of maximual value of absolute response
    max_go_index = np.argmax(np.abs(mean_go_compreg),axis = 1).reshape((1,-1))[0]
    max_nogo_index = np.argmax(np.abs(mean_nogo_compreg),axis = 1).reshape((1,-1))[0]
    
    magnitude_go = np.array([mean_go_compreg[i,k] for i,k in enumerate(max_go_index)])
    magnitude_nogo = np.array([mean_nogo_compreg[i,k] for i,k in enumerate(max_nogo_index)])
    
    # activated neurons
    mean_go_act = mean_go[magnitude_go>0.5,:]
    mean_nogo_act = mean_nogo[magnitude_nogo>0.3,:]
    # inhibited neurons
    mean_go_inh = mean_go[magnitude_go <-0.5,:]
    mean_nogo_inh = mean_nogo[magnitude_nogo <-0.3,:]
    #mean of subgroup
    mean_sub_go_act = np.nanmean(mean_go_act,axis = 0)
    std_sub_go_act = np.nanstd(mean_go_act,axis = 0)/mean_go_act.shape[1]
    mean_sub_nogo_act = np.nanmean(mean_nogo_act,axis = 0)
    std_sub_nogo_act = np.nanstd(mean_nogo_act,axis = 0)/mean_nogo_act.shape[1]
    mean_sub_go_inh = np.nanmean(mean_go_inh,axis = 0)
    std_sub_go_inh = np.nanstd(mean_go_inh,axis = 0)/mean_go_inh.shape[1]
    mean_sub_nogo_inh = np.nanmean(mean_nogo_inh,axis = 0)
    std_sub_nogo_inh = np.nanstd(mean_nogo_inh,axis = 0)/mean_nogo_inh.shape[1]

    # figure plot        
    colormap = "coolwarm"
    cmap = mpl.cm.get_cmap(colormap) 
 
    x = np.linspace(0,(len(mean_sub_go_act)-1)/5,len(mean_sub_go_act))
    color = cmap(index/len(sessions))
    # axes[index].plot(x,mean_sub_go_act, lw=3,  color = color, label = 'session {}'.format(sessions[index]),alpha=0.8)
    # axes[index].fill_between(x,mean_sub_go_act+std_sub_go_act,mean_sub_go_act-std_sub_go_act, facecolor= color, alpha=0.4) 
    # axes[index].plot(x,mean_sub_go_inh, lw=3,  color = color, alpha=0.8)
    # axes[index].fill_between(x,mean_sub_go_inh+std_sub_go_inh,mean_sub_go_inh-std_sub_go_inh, facecolor= color, alpha=0.4) 
    axes[index].plot(x,mean_sub_nogo_act, lw=3,  color = color, label = 'session {}'.format(sessions[index]),alpha=0.8)
    axes[index].fill_between(x,mean_sub_nogo_act+std_sub_nogo_act,mean_sub_nogo_act-std_sub_nogo_act, facecolor= color, alpha=0.4) 
    axes[index].plot(x,mean_sub_nogo_inh, lw=3,  color = color, alpha=0.8)
    axes[index].fill_between(x,mean_sub_nogo_inh+std_sub_nogo_inh,mean_sub_nogo_inh-std_sub_nogo_inh, facecolor= color, alpha=0.4) 
    
    axes[index].axhline(y=0,color = 'k',alpha = 0.5) 
    axes[index].set_ylim(-1.2,1.5)
    
    axes[index].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False) 
    axes[index].spines["top"].set_visible(False)    
    axes[index].spines["right"].set_visible(False)
    axes[index].spines["bottom"].set_visible(False )    
    axes[index].spines["left"].set_visible(False )    
    axes[index].set_xlabel('Time (s)' )
    axes[index].set_ylabel('Z-score' if index == 0 else '')
    axes[index].fill_between([1,2], [10,10],[-10,-10],facecolor="grey", color='grey',alpha=0.2)
    axes[index].set_title('Session {} nogo'.format(sessions[index]))

savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/separate_nogo_longer_mean_activities_activated_inhibited_neurons'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()


















