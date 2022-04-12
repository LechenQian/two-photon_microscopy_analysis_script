# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:10:22 2020

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
import seaborn as sns


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
for im_name in templates_name_3:
    temp = io.imread(im_name)
    vmax = temp.max()
    temp = temp/vmax
    templates.append(temp)
# dimension of the images
dims = templates[0].shape
#%%
# spatial_union, assignments, matchings = register_multisession(A=spatial, dims=dims, templates=templates)
select_session = [5,6,7]
spatial_sel = [x for i,x in enumerate(spatial) if i in select_session]
temp_sel = [x for i,x in enumerate(templates) if i in select_session]
spatial_union, assignments, matchings = register_multisession(A=spatial_sel, dims=dims, templates=temp_sel)
order = np.argsort(np.sum(np.isnan(assignments),axis = 1))
assignments_new = assignments[order,:]
assignments = assignments_new.copy()
assignments
#%%
n_reg = 3  # minimal number of sessions that each component has to be registered in

# Use number of non-NaNs in each row to filter out components that were not registered in enough sessions 
# (selina: confirmed, a row is a specific component in assignment matrix. and matching can tell you the kth component in J session is the ith component.)
assignments_filtered = np.array(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg], dtype=int);
trialnum = []
for i in range(assignments_filtered.shape[1]):
    # temp_mat = big_water_list[select_session[i]]
    temp_mat = big_unpwater_list[select_session[i]-5]
    sel_mat = temp_mat[assignments_filtered[:,i],:,:]
    if i == 0:
        tca_mat = sel_mat
    else:
        tca_mat = np.concatenate([tca_mat,sel_mat],axis = 2)
    trialnum.append(temp_mat.shape[2])
print(tca_mat.shape)  
# convert NaN to zero
tca_mat = np.nan_to_num(tca_mat)
tca_mat = tca_mat+5  
#%%
import tensortools as tt



# Fit CP tensor decomposition (two times).
rank = 3

U = tt.ncp_hals(tca_mat, rank=rank, verbose=True)
# V = tt.ncp_hals(tca_mat, rank=rank, verbose=True)

# Compare the low-dimensional factors from the two fits.
cum_trialnum = np.cumsum(trialnum)
cum_trialnum = np.insert(cum_trialnum,0,0)
fig, _= plot_factors(U.factors,cum_trialnum,plots=['bar','line','scatter'])
# plot_factors(V.factors, cum_trialnum,plots=['bar','line','scatter'])


# Show plots.
plt.show()
#%%
from matplotlib.cm import get_cmap
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 
fig,axes = plt.subplots(rank,4,figsize=(20,2.7*rank),gridspec_kw={'width_ratios': [1,1,3, 1]})

for r in range(rank):
    for i, f in enumerate(U.factors):
        x = np.arange(0, f.shape[0])
        if i == 0:
            axes[r, i].bar(x, f[:, r],color = 'grey',width = 1,edgecolor = 'k' )
            axes[r, i].set_xlim(-0.2, f.shape[0]+1)
        elif i== 1:
            axes[r, i].plot(f[:, r]-np.mean(f[:3, r]), '-',color = 'k',linewidth = 3)
            axes[r, i].set_xlim(-0.2, f.shape[0])
        elif i == 2:
            for k in range(len(cum_trialnum)-1):
                x = np.arange(cum_trialnum[k],cum_trialnum[k+1],1)
                color = cmap(k/(len(cum_trialnum)-1))
                axes[r,i].scatter(x, f[cum_trialnum[k]:cum_trialnum[k+1], r],c = color,s = 5)
                axes[r,i+1].bar(k, np.mean(f[cum_trialnum[k]:cum_trialnum[k+1], r]),color = color,width = 1,
                                edgecolor = 'k',yerr=np.std(f[cum_trialnum[k]:cum_trialnum[k+1], r]))#/(cum_trialnum[k+1]-cum_trialnum[k]))#,s = 15)
            axes[r, i+1].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False)
            axes[r, i+1].spines["top"].set_visible(False)    
            axes[r, i+1].spines["right"].set_visible(False)
            axes[r, i+1].set_xticks(np.arange(0,len(cum_trialnum)-1)) 
            axes[r, i+1].set_xticklabels(np.arange(1,len(cum_trialnum)))
                
        axes[r, i].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False)
        axes[r, i].spines["top"].set_visible(False)    
        axes[r, i].spines["right"].set_visible(False)
for r in range(U.factors.rank):
   for i in range(1,U.factors.ndim,1):
       # only two labels
       
       ymin, ymax = np.round(axes[r, i].get_ylim(), 1)
       # if i ==2 :
       #     ymin = -2.0
       #     ymax = 2.0
       axes[r, i].set_ylim((ymin-0.1, ymax+0.1))

       # remove decimals from labels
       if ymin.is_integer():
           ymin = int(ymin)
       if ymax.is_integer():
           ymax = int(ymax)
       
       # update plot
       axes[r, i].set_yticks([ymin-0.1, 0,ymax+0.1])           
savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/tca_6810_unpredicted_water_trials_1'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 200,transparent = True)
plt.show()            
#%% mean_tca_mat
n_reg = 9  # minimal number of sessions that each component has to be registered in

# Use number of non-NaNs in each row to filter out components that were not registered in enough sessions 
# (selina: confirmed, a row is a specific component in assignment matrix. and matching can tell you the kth component in J session is the ith component.)
assignments_filtered = np.array(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg], dtype=int);
trialnum = []
for i in range(assignments_filtered.shape[1]):
    temp_mat = big_goodor_list[i]
    sel_mat = np.nanmean(temp_mat[assignments_filtered[:,i],:,:],axis = 2)
    if i == 0:
        tca_mat = sel_mat
    else:
        tca_mat = np.dstack([tca_mat,sel_mat])
    trialnum.append(1)
print(tca_mat.shape)    
#%%
import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# convert NaN to zero
tca_mat = np.nan_to_num(tca_mat)

# Fit CP tensor decomposition (two times).
rank = 4
U = tt.cp_als(tca_mat, rank=rank, verbose=True)
V = tt.cp_als(tca_mat, rank=rank, verbose=True)

# Compare the low-dimensional factors from the two fits.
cum_trialnum = np.cumsum(trialnum)[0:-1]
cum_trialnum = np.insert(cum_trialnum,0,0)

fig, _= plot_factors(U.factors,cum_trialnum,plots=['bar','line','scatter'])
# tt.plot_factors(V.factors, fig=fig)


# Show plots.
plt.show()
#%%
from matplotlib.cm import get_cmap
colormap = "coolwarm"
cmap = mpl.cm.get_cmap(colormap) 
fig,axes = plt.subplots(rank,4,figsize=(20,15))
for r in range(rank):
    for i, f in enumerate(U.factors):
        x = np.arange(0, f.shape[0])
        if i == 0:
            axes[r, i].bar(x, f[:, r])
            axes[r, i].set_xlim(-0.2, f.shape[0]+1)
        elif i== 1:
            axes[r, i].plot(f[:, r], '-')
            axes[r, i].set_xlim(-0.2, f.shape[0])
        elif i == 2:
            for k in range(len(cum_trialnum)-1):
                x = np.arange(cum_trialnum[k],cum_trialnum[k+1],1)
                color = cmap(k/(len(cum_trialnum)-1))
                axes[r,i].scatter(x, f[cum_trialnum[k]:cum_trialnum[k+1], r],c = color,s = 2)
                axes[r,i+1].scatter(cum_trialnum[k]/2+cum_trialnum[k+1]/2, np.mean(f[cum_trialnum[k]:cum_trialnum[k+1], r]),c = color,s = 15)
                axes[r, i+1].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False)
                axes[r, i+1].spines["top"].set_visible(False)    
                axes[r, i+1].spines["right"].set_visible(False)
                
        axes[r, i].tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False)
        axes[r, i].spines["top"].set_visible(False)    
        axes[r, i].spines["right"].set_visible(False)

savename = 'D:/PhD/Microscope/Selina/imaging_data/new_figures/{}-{}/tca_mean_whole_go_trials_0'.format(mouse_id,date)
plt.savefig(savename+'.png', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.eps', bbox_inches="tight", dpi = 200,transparent = True)
plt.savefig(savename+'.svg', bbox_inches="tight", dpi = 200,transparent = True)
plt.show()            
#%%
def plot_factors(U, cum_trialnum, plots='line',**kwargs):
    """Plots a KTensor.

    Note: Each keyword option is broadcast to all modes of the KTensor. For
    example, if `U` is a 3rd-order tensor (i.e. `U.ndim == 3`) then
    `plot_factors(U, plots=['line','bar','scatter'])` plots all factors for the
    first mode as a line plot, the second as a bar plot, and the third mode as
    a scatterplot. But, thanks to broadcasting semantics,
    `plot_factors(U, color='line')` produces line plots for each mode.

    Parameters
    ----------
    U : KTensor
        Kruskal tensor to be plotted.

    plots : str or list
        One of {'bar','line','scatter'} to specify the type of plot for each
        factor. The default is 'line'.
    fig : matplotlib Figure object
        If provided, add plots to the specified figure. The figure must have a
        sufficient number of axes objects.
    axes : 2d numpy array of matplotlib Axes objects
        If provided, add plots to the specified figure.
    scatter_kw : dict or sequence of dicts
        Keyword arguments provided to scatterplots. If a single dict is
        provided, these options are broadcasted to all modes.
    line_kw : dict or sequence of dicts
        Keyword arguments provided to line plots. If a single dict is provided,
        these options are broadcasted to all modes.
    bar_kw : dict or sequence of dicts
        Keyword arguments provided to bar plots. If a single dict is provided,
        these options are broadcasted to all modes.
    **kwargs : dict
        Additional keyword parameters are passed to the `subplots(...)`
        function to specify options such as `figsize` and `gridspec_kw`. See
        `matplotlib.pyplot.subplots(...)` documentation for more info.
    """

    # ~~~~~~~~~~~~~
    # PARSE OPTIONS
    # ~~~~~~~~~~~~~
    kwargs.setdefault('figsize', (8, U.rank))
    from matplotlib.cm import get_cmap
    colormap = "coolwarm"
    cmap = mpl.cm.get_cmap(colormap) 

    
    
    fig, axes = plt.subplots(U.rank, U.ndim, **kwargs)
    # make sure axes is a 2d-array
    if U.rank == 1:
        axes = axes[None, :]

    # main loop, plot each factor
    plot_obj = np.empty((U.rank, U.ndim), dtype=object)
    for r in range(U.rank):
        for i, f in enumerate(U):
            # start plots at 1 instead of zero
            x = np.arange(1, f.shape[0]+1)

            # determine type of plot
            if plots[i] == 'bar':
                axes[r, i].bar(x, f[:, r])
                axes[r, i].set_xlim(-0.2, f.shape[0]+1)
            elif plots[i] == 'scatter':
                for k in range(len(cum_trialnum)-1):
                    x = np.arange(cum_trialnum[k],cum_trialnum[k+1],1)
                    color = cmap(k/(len(cum_trialnum)-1))
                    axes[r, i].scatter(x, f[cum_trialnum[k]:cum_trialnum[k+1], r],c = color,s = 4)
                axes[r, i].set_xlim(-0.2, f.shape[0])
            elif plots[i] == 'line':
                axes[r, i].plot(f[:, r], '-')
                axes[r, i].set_xlim(-0.2, f.shape[0])
            else:
                raise ValueError('invalid plot type')

            # format axes
            axes[r, i].locator_params(nbins=4)
            axes[r, i].spines['top'].set_visible(False)
            axes[r, i].spines['right'].set_visible(False)
            axes[r, i].xaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_ticks_position('left')
            axes[r, i].xaxis.set_ticks_position('bottom')

            # remove xticks on all but bottom row
            if r != U.rank-1:
                plt.setp(axes[r, i].get_xticklabels(), visible=False)

    # # link y-axes within columns
    # for i in range(U.ndim):
    #     yl = [a.get_ylim() for a in axes[:, i]]
    #     y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
    #     [a.set_ylim((y0, y1)) for a in axes[:, i]]

    # format y-ticks
    for r in range(U.rank):
        for i in range(U.ndim):
            # only two labels
            
            ymin, ymax = np.round(axes[r, i].get_ylim(), 1)
            if i ==2 :
                ymin = -2.0
                ymax = 2.0
            axes[r, i].set_ylim((ymin, ymax))

            # remove decimals from labels
            if ymin.is_integer():
                ymin = int(ymin)
            if ymax.is_integer():
                ymax = int(ymax)
            
            # update plot
            axes[r, i].set_yticks([ymin, ymax])

    plt.tight_layout()

    return fig, axes


#%%
def pickle_dict(df,path,filename):
    try:
        os.makedirs(path) # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path +'/{}.pickle'.format(filename)
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save to pickle done!')
    
dict = {'spatial_footprints':spatial,'normalized_templates':templates,'dimension':dims}
savepath = 'D:/PhD/to_fred'
pickle_dict(dict,savepath,'neurons_to_register')