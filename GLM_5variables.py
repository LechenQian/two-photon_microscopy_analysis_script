# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:30:37 2020

@author: qianl
"""


#%% import packages
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
from statsmodels.gam.api import GLMGam
import statsmodels.gam.smooth_basis as sb
import statsmodels.api as sm
from patsy import dmatrices
#%% import functions

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
def bin_neural_signal(dff_signal_one_neuron,row_df,binsize = 1):
    binned_dff = []
    odor_on = row_df['solenoid_switch_frame'][0]

    bins = np.linspace(-1,5,7) ## need to adjusted for bins
    for i in bins:
        chunk_dff = dff_signal_one_neuron[max(int(odor_on+i*binsize),0):min(int(odor_on+(i+1)*binsize),row_df['frames'])]
#         if i == -1:
#             mean = np.mean(chunk_dff)
#         chunk_dff = chunk_dff-mean

        value_dff = np.mean(chunk_dff)
        binned_dff.append(value_dff)

    return binned_dff

def bin_licking(row_df,trialtype,binsize = 1):
    
    if trialtype in ['go','go_omit']:
        odor_on = row_df['go_odor_on'][0]
    elif trialtype == 'no_go':
        odor_on = row_df['nogo_odor_on'][0]
    else:
        print('wrong backgournd trial type')
    binned_licking = []
    bins = np.linspace(-1,5,7) ## need to adjusted for bins
    for i in bins:
        chunk_licking = [x for x in row_df['licking'] if x >=int(odor_on+i*binsize)and x<int(odor_on+(i+1)*binsize)]
        value_licking = len(chunk_licking)/binsize
        binned_licking.append(value_licking)
    return binned_licking
def get_r2(y,model):
    sst_val = sum(map(lambda x: np.power(x,2),y.values-np.mean(y.values))) 
    sse_val = sum(map(lambda x: np.power(x,2),model.resid_response)) 
    r2 = 1.0 - sse_val/sst_val
    return r2[0]
#%% import data
mouse_id  ='OT-GC-1'
date = '0612'
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
#%% calculate z-score and select responsive neurons
mat_go_allses = []
mat_nogo_allses = []
mat_goomit_allses = []
res_neuron_allses = []

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
        # absolute magnitude
        magnitude_go = np.array([np.abs(mean_go_compreg)[i,k] for i,k in enumerate(max_go_index)])
        magnitude_nogo = np.array([np.abs(mean_nogo_compreg)[i,k] for i,k in enumerate(max_nogo_index)])
        # index
        index_go = [i for i,k in enumerate(magnitude_go) if k>0.5]
        index_nogo = [i for i,k in enumerate(magnitude_nogo) if k>0.5]
        # union of max_go_index and max_nogo_index
        index_union = np.union1d(index_go, index_nogo)
        # replace all nan with zeros
        mat_go_allses.append(np.nan_to_num(F_dff_matrix1[:,int(odor_on_frame1-5):int(odor_on_frame1+30)]))
        mat_goomit_allses.append(np.nan_to_num(F_dff_matrix2[:,int(odor_on_frame2-5):int(odor_on_frame2+30)]))
        mat_nogo_allses.append(np.nan_to_num(F_dff_matrix3[:,int(odor_on_frame3-5):int(odor_on_frame3+30)]))
        res_neuron_allses.append(index_union)

#%% create GLM dataset
session_index= 2
use_neuron_index = np.arange(signals[session_index].shape[0])#res_neuron_allses[session_index] # np.arange(signals[session_index].shape[0])
num_neuron = len(use_neuron_index)
# num_neuron = signals[session_index].shape[0]
rel_contrib_mat = np.zeros([num_neuron,5])
for index,neuron_index in enumerate(use_neuron_index):
    big_dict = Dicts[session_index]
    df_all_drop = big_dict['behavior_dropped_df']   
    df_original = big_dict['behavior_original_df']
    
    binned_licking = []
    binned_dff = []
    diff_binned_licking = []
    trialnum = []
    water = np.zeros([7,7])
    go_odor = np.zeros([7,7])
    nogo_odor = np.zeros([7,7])
    
    GLM_df = pd.DataFrame()
    denoised_individual_trials = slice_signal_to_trial(signals[session_index],df_original)   
    
    neuron_baseline_std = std_in_baseline(num_neuron,'background',denoised_individual_trials,big_dict['dict_index_trialtype'])
    std = neuron_baseline_std[neuron_index]
    for i, row in df_all_drop.iterrows():
        
        if row['trialtype'] != 'background':
            temp = list(bin_neural_signal(denoised_individual_trials['Trial{}'.format(i)][neuron_index,:],row_df=row,binsize = 5)/std)
            binned_dff += temp # calculate z-score
            binned_licking += bin_licking(row_df=row,trialtype=row['trialtype'],binsize = 1)
            if len(binned_dff) != len(binned_licking):
                print(len(binned_dff),len(binned_licking))
                print('bin number doesn\'t match.')
            diff_licking = np.diff(bin_licking(row_df=row,trialtype=row['trialtype'],binsize = 1))
            diff_licking = np.insert(diff_licking,0,0)
            diff_binned_licking += list(diff_licking)
            
            # whole-trial variable
            trialnum += [i]*7
            # events variable
            x = np.linspace(0,6,7)
            knots = sb.get_knots_bsplines(x,df = 7,)#spacing = 'equal')
    
            basis = sb._eval_bspline_basis(x,degree = 3,knots=knots)[0]
            temp_water = np.zeros([7,7])
            temp_go_odor = np.zeros([7,7])
            temp_nogo_odor = np.zeros([7,7])
            if row['trialtype'] == 'go':
                water_event = np.array([0,0,0,0,1,0,0])
                go_odor_event = np.array([0,1,0,0,0,0,0])
                for i in range(7):
                    conv_water = np.convolve(basis[:,i],water_event,mode = 'full')
                    temp_water[i,:] = conv_water[0:7]
                    conv_go = np.convolve(basis[:,i],go_odor_event,mode = 'full')
                    temp_go_odor[i,:] = conv_go[0:7]
    
            elif row['trialtype'] == 'go_omit':
                go_odor_event = np.array([0,1,0,0,0,0,0])
                for i in range(7):
                    conv_go = np.convolve(basis[:,i],go_odor_event,mode = 'full')
                    temp_go_odor[i,:] = conv_go[0:7]
                
            elif row['trialtype'] == 'no_go':
                water_event = np.array([0,0,0,0,0,0,0])
                nogo_odor_event = np.array([0,1,0,0,0,0,0])
                for i in range(7):
                    conv_water = np.convolve(basis[:,i],water_event,mode = 'full')
                    temp_water[i,:] = conv_water[0:7]
                    conv_nogo = np.convolve(basis[:,i],nogo_odor_event,mode = 'full')
                    temp_nogo_odor[i,:] = conv_nogo[0:7]
    
            water = np.hstack((water,temp_water))
            go_odor = np.hstack((go_odor,temp_go_odor))
            nogo_odor = np.hstack((nogo_odor,temp_nogo_odor))
    
    water = water[:,7:]
    go_odor = go_odor[:,7:]
    nogo_odor = nogo_odor[:,7:]
    GLM_df = pd.DataFrame({'NeuroSignal_dff':binned_dff,'licking':binned_licking,'trialnum':trialnum,'diff_licking':diff_binned_licking,'square_licking':np.power(binned_licking,2)})
    
    for i in range(7):
        GLM_df['water{}'.format(i)] = water[i,:]
    for i in range(7):
        GLM_df['go_odor{}'.format(i)] = go_odor[i,:]    
    for i in range(7):
        GLM_df['nogo_odor{}'.format(i)] = nogo_odor[i,:]    
    GLM_df.describe()
    #  1. loop neuron 2. train set and test set (function) 3. fit all variables and partial model, generate R2 4. calculate the relative contribution and build the matr
    formula = 'NeuroSignal_dff ~ licking + diff_licking + square_licking + trialnum + go_odor0 + go_odor1 + go_odor2+go_odor3 + go_odor4 + go_odor5 + go_odor6 + water0+ water1+ water2+ water3+ water4+ water5+ water6+ nogo_odor0+ nogo_odor1+ nogo_odor2+ nogo_odor3 + nogo_odor4+ nogo_odor5 + nogo_odor6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o = glm.fit()
    r2= get_r2(y,res_o)
    
    # no licking
    formula = 'NeuroSignal_dff ~ trialnum + go_odor0 + go_odor1 + go_odor2+go_odor3 + go_odor4 + go_odor5 + go_odor6 + water0+ water1+ water2+ water3+ water4+ water5+ water6+ nogo_odor0+ nogo_odor1+ nogo_odor2+ nogo_odor3 + nogo_odor4+ nogo_odor5 + nogo_odor6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o_licking = glm.fit()
    r2_licking= get_r2(y,res_o_licking)
    
    # no go odor
    formula = 'NeuroSignal_dff ~ licking + diff_licking + square_licking + trialnum + water0+ water1+ water2+ water3+ water4+ water5+ water6+ nogo_odor0+ nogo_odor1++ nogo_odor2+ nogo_odor3 + nogo_odor4+ nogo_odor5 + nogo_odor6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o_go = glm.fit()
    r2_go_odor= get_r2(y,res_o_go)
    
    # no water
    formula = 'NeuroSignal_dff ~ licking+ diff_licking + square_licking + trialnum + go_odor0 + go_odor1 + go_odor2+go_odor3 + go_odor4 + go_odor5 + go_odor6 + nogo_odor0+ nogo_odor1+ nogo_odor2+ nogo_odor3 + nogo_odor4+ nogo_odor5 + nogo_odor6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o_water = glm.fit()
    r2_water= get_r2(y,res_o_water)
    
    # no nogo odor
    formula = 'NeuroSignal_dff ~ licking+ diff_licking + square_licking + trialnum + go_odor0 + go_odor1 + go_odor2+go_odor3 + go_odor4 + go_odor5 + go_odor6 + water0+ water1+ water2+ water3+ water4+ water5+ water6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o_nogo = glm.fit()
    r2_nogo_odor= get_r2(y,res_o_nogo)
    
    # no trialnum
    formula = 'NeuroSignal_dff ~ licking+ diff_licking + square_licking + go_odor0 + go_odor1 + go_odor2+go_odor3 + go_odor4 + go_odor5 + go_odor6 + water0+ water1+ water2+ water3+ water4+ water5+ water6+ nogo_odor0+ nogo_odor1++ nogo_odor2+ nogo_odor3 + nogo_odor4+ nogo_odor5 + nogo_odor6'
    y, X = dmatrices(formula, data=GLM_df, return_type='dataframe')
    glm = sm.GLM(y,X)
    res_o_trialnum = glm.fit()
    r2_trialnum= get_r2(y,res_o_trialnum)
    
    rel_contrib = (r2-r2_go_odor)/((r2-r2_water)+(r2-r2_licking)+(r2-r2_go_odor)+(r2-r2_nogo_odor)+(r2-r2_trialnum))
    print('relative contribution:',rel_contrib)
    rel_contrib_mat[index,0] = rel_contrib
    
    rel_contrib = (r2-r2_nogo_odor)/((r2-r2_water)+(r2-r2_licking)+(r2-r2_go_odor)+(r2-r2_nogo_odor)+(r2-r2_trialnum))
    print('relative contribution:',rel_contrib)
    rel_contrib_mat[index,1] = rel_contrib
    
    rel_contrib = (r2-r2_water)/((r2-r2_water)+(r2-r2_licking)+(r2-r2_go_odor)+(r2-r2_nogo_odor)+(r2-r2_trialnum))
    print('relative contribution:',rel_contrib)
    rel_contrib_mat[index,2] = rel_contrib
    
    rel_contrib = (r2-r2_licking)/((r2-r2_water)+(r2-r2_licking)+(r2-r2_go_odor)+(r2-r2_nogo_odor)+(r2-r2_trialnum))
    print('relative contribution:',rel_contrib)
    rel_contrib_mat[index,3] = rel_contrib
    
    rel_contrib = (r2-r2_trialnum)/((r2-r2_water)+(r2-r2_licking)+(r2-r2_go_odor)+(r2-r2_nogo_odor)+(r2-r2_trialnum))
    print('relative contribution:',rel_contrib)
    rel_contrib_mat[index,4] = rel_contrib
#
mean_var = np.mean(rel_contrib_mat,axis = 0)
std_var = np.std(rel_contrib_mat,axis = 0)
fig, ax = plt.subplots(figsize = (5,4))
x = [1,2,3,4,5]
# ax.bar(x, mean_var, yerr=std_var/2, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.boxplot(rel_contrib_mat)
ax.set_ylabel('relatie contribution (%)')
ax.set_xticks(x)
ax.set_xticklabels(['go odor','no-go odor','water','licking','trialnum'],rotation = 'vertical')
ax.set_title('Session {}'.format(sessions[session_index]),pad = 20)
ax.set_ylim([0,1])
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/relative_contribution_variables_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)

plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()
#
fig, ax = plt.subplots(1,5,figsize = (8,3),sharey = True)
xlabels = ['go odor','no-go odor','water','licking','trialnum']
bin_edge = np.linspace(0,1,20)
yticks = np.arange(0,num_neuron,10)
for i in range(5):

    ax[i].hist(rel_contrib_mat[:,i],bins = bin_edge,alpha=0.5, edgecolor='black')
    
    ax[i].set_xticks([0,1])
    ax[i].set_xticklabels([0,100])
    ax[i].set_yticks(yticks)
    ax[i].set_title(xlabels[i])
    
    
    ax[i].yaxis.grid(True)

# Save the figure and show
fig.text(0.5, 0.001, 'Relative contribution (%)', ha='center')
fig.text(0.001, 0.5, 'Neurons', va='center', rotation='vertical')
plt.tight_layout()
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/relative_contribution_variables_histogram_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)

plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()
#
yticks = np.arange(0,num_neuron,10)
ax = sns.heatmap(rel_contrib_mat)
ax.set_ylabel('Neurons')
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
ax.set_xticks(np.array(x)-0.5)
ax.set_xticklabels(['go odor','no-go odor','water','licking','trialnum'],rotation = 'vertical')
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/relative_contribution_variables_heatmap_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)

plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()

#%% GMM clustering
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

X = rel_contrib_mat
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['full']#['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['turquoise'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(4,5))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i)
    bars.append(plt.bar(xpos, bic,
                        width=.5, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max() * 0.97])
plt.title('BIC score per model',pad = 20)
print(xpos)
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/BIC_CLUSTER_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()
Y_ = clf.predict(X)
# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)
# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()
cluster0 = [i for i,v in enumerate(Y_) if v == 0]
cluster1 = [i for i,v in enumerate(Y_) if v == 1]
cluster2 = [i for i,v in enumerate(Y_) if v == 2]

cluster0_mat = rel_contrib_mat[cluster0,:]
cluster1_mat = rel_contrib_mat[cluster1,:]
cluster2_mat = rel_contrib_mat[cluster2,:]
new_rel_contrib_mat = np.vstack((cluster0_mat,cluster1_mat,cluster2_mat))
sns.heatmap(new_rel_contrib_mat)
plt.axhline(y = len(cluster0),color = 'w')
plt.axhline(y = len(cluster0)+len(cluster1),color = 'w')
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/unsorted_relative_contribution_gmm_clusters_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
#%%


cluster0 = [i for i,v in enumerate(Y_) if v == 0]
cluster1 = [i for i,v in enumerate(Y_) if v == 1]
cluster2 = [i for i,v in enumerate(Y_) if v == 2]

cluster0_mat = rel_contrib_mat[cluster0,:]
cluster1_mat = rel_contrib_mat[cluster1,:]
cluster2_mat = rel_contrib_mat[cluster2,:]

s_index = np.argsort(cluster0_mat[:,4])
cluster0_mat = cluster0_mat[s_index,:]
s_index = np.argsort(cluster1_mat[:,0])
cluster1_mat = cluster1_mat[s_index,:]
s_index = np.argsort(cluster2_mat[:,3])
cluster2_mat = cluster2_mat[s_index,:]

new_rel_contrib_mat = np.vstack((cluster0_mat,cluster1_mat,cluster2_mat))

sns.heatmap(new_rel_contrib_mat)
plt.axhline(y = len(cluster0),color = 'w')
plt.axhline(y = len(cluster0)+len(cluster1),color = 'w')
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/relative_contribution_gmm_clusters_session{}'.format(mouse_id,date,sessions[session_index])
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 400,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 400,transparent = True)
plt.show()