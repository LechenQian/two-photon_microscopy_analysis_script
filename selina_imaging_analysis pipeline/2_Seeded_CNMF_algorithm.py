# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:30:09 2020

@author: qianl


"""
#%% import functions
import bokeh.plotting as bpl
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
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
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
import os
import pandas as pd
import re
import math
import seaborn as sns
#%% basic operations
def read_image_file(filedir):
    
    
    
    filenames  = []
    file_index = []
    file_len   = []
    
    for dirpath, dirnames, files in os.walk(filedir): # can walk through all levels down
    #     print(f'Found directory: {dirpath}')
        for f_name in files:
            if f_name.endswith('.tiff'):
                filename = dirpath+'/'+f_name
                try:
                    index = list(map(int,re.findall(r"(\d{1,3})",f_name)))[1]
                    im = io.imread(filename)
                    len_t = np.min(im.shape)
                    if len_t == 200:
                        len_t = 1

                    filenames.append(filename)
                    file_index.append(int(index))
                    file_len.append(len_t)
                    file_date = re.search(r"(\d{1,3})",dirpath).group(0)
                except:
                    pass
    reindex = np.argsort(file_index)
    sort_file_index = np.sort(file_index)
    sort_file_len = [file_len[i] for i in reindex]
    sort_filenames = [filenames[i] for i in reindex]
    print('---------------------------------------------')    
    print('Files have been loaded.')
    return sort_file_index, sort_file_len, sort_filenames
import pickle
import sys
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
#%% read imaging data
# 正则读取有多少的符合规则的文件
#################### change
date = '2020-09-19'
mouse_id = 'OT-GC-9-D1'

filedir = 'D:/PhD/Microscope/imaging_data_20_09/D1_MSN/{}/{}'.format(mouse_id,date)


file_index, file_frame, filenames = read_image_file(filedir)

index = pd.Series(file_index, name = 'file_index')
frame = pd.Series(file_frame, name = 'frames')
file_name = pd.Series(filenames, name = 'filename')
df = pd.concat([index,frame, file_name], axis=1)
#%% adding behavioral part
sys.path.insert(0,'D:/PhD/Behavior/Behavior_Analysis/2020_batch_clean_selected')
from parse_data_2020 import Mouse_data
from parse_data_2020 import pickle_dict
from parse_data_2020 import load_pickleddata
#################### change

load_path = 'D:/PhD/Behavior/behavior_20_09/experiment_data_2020_09_Pav_imaging/parsed_dataframe_pickle/{0}_stats.pickle'.format(mouse_id)
mouse = load_pickleddata(load_path)

#%% load mmap images to python format
# memory map the file in order 'C'
savedir = 'D:/PhD/Microscope/imaging_data_20_09/D1_MSN/{}/processed/{}'.format(mouse_id,date)
stack_name = [savedir+'/tiff_chained_all_corrected.tiff']
loaddir = savedir
fname_new = '{}/memmap__d1_201_d2_200_d3_1_order_C_frames_{}_.mmap'.format(loaddir,sum(df.frames.values))
Yr, dims, T = cm.load_memmap(fname_new)

# this images will be used for CNMF algorithm
images = np.reshape(Yr.T, [T] + list(dims), order='F') #load frames in python format (T x X x Y)

#%% CNMF data dependent parameters
# dataset dependent parameters
fr = 5                            # imaging rate in frames per second
decay_time = 1                    # length of a typical transient in seconds  #jGCaMP7s sensor Half decay time (ms) 1260±40

    
# motion correction parameters
strides = (40, 40)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (10, 10)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (5,5)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                    # number of global background components
merge_thr = 0.85          # merging threshold, max correlation allowed
stride_cnmf = 5             # amount of overlap between the patches in pixels
K = 5                       # number of components per patch
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization
rf = None                   # half-size of the patches in pixels. Should be `None` when seeded CNMF is used.
only_init = False           # has to be `False` when seeded CNMF is used
gSig = (3, 3)               # expected half size of neurons in pixels, very important for proper component detection


# params object
opts_dict = {'fnames': stack_name,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': only_init,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr, 
            }

opts = params.CNMFParams(params_dict=opts_dict)

#%% # load binary mask
'''
save this binary mask to the same direcotry as mmap images. name them as bmask
'''
#invert binary mask 
path = loaddir
print(path)
mask = 1+(-1)*np.asarray(imread('{}/Mask_USE.png'.format(path)), dtype=bool)

# calculate distances from nearest edge
distances = ndi.distance_transform_edt(mask)

# apply threshold of expected cell diameter gSig_seed to get one maximum per cell
gSig_seed = 2
local_max = distances.copy()
local_max[local_max >= gSig_seed/2] = gSig_seed
local_max[local_max < gSig_seed/2] = 0
local_max = local_max.astype('bool')

#Separate components with watershed algorithm
# generate labels of isolated maxima
markers = ndi.label(local_max)[0]
# remove very small features (avoids irregular cells being counted multiple times)
sizes = np.bincount(markers.ravel())      # get list of number of pixels of each label
mask_sizes = sizes > 0                   # remove labels with very low pixel counts
mask_sizes[0] = 0                         # remove count of background label
local_max_cleaned = mask_sizes[markers]   # apply mask to binary image to only keep large components
# update labels with cleaned-up features
markers_cleaned = ndi.label(local_max_cleaned)[0]
# apply watershed to the original binary mask using the cleaned labels to create separated ROIs
labels = watershed(-distances, markers_cleaned, mask=mask)
# things to saved here: 1. watershed images with seperate labels
# visualize separated labels after watershed (each feature has a slightly different color)
plt.imshow(labels,cmap=plt.cm.nipy_spectral)
plt.title('Labels of features separated by watershed algorithm')
plt.savefig("{}/watershed.png".format(path), bbox_inches="tight", dpi = 400)
plt.show()

# Bring labels into sparse column format that CNMF can use
'''
Now we have each component marked with a different label, even if their ROIs are overlapping or merged. We can use this matrix to bring it into the sparse column format that the CNMF uses to store spatial components. Each of the labelled features will be extracted separately and used as the spatial component for this neuron.
'''
num_features = np.max(labels)
A = np.zeros((labels.size, num_features), dtype=bool)

# save each component individually into the sparse column matrix
for i in range(num_features):
    temp = (labels == i + 1)                     # each feature is saved as a single component in its own frame
    temp = dilation(temp, selem=np.ones((3,3)))  # dilate spatial component to increase error margin a bit

    # parse the current component 'temp' into the sparse column matrix
    A[:, i] = temp.flatten('F')

manual_mask = A.copy()

#%% Run mask seeded CNMF
'''
rf = None # half-size of the patches in pixels. Should be None when seeded CNMF is used. only_init = False # has to be False when seeded CNMF is used gSig = (3, 3) # expected half size of neurons in pixels, very important for proper component detection
'''
cnm = cnmf.CNMF(n_processes=8, params=opts, Ain = manual_mask)
cnm = cnm.fit(images)

#%% plot contours of found components
Cns = local_correlations_movie_offline(fname_new,#mc.mmap_file[0],
                                       remove_baseline=True, window=1000, stride=1000,
                                       winSize_baseline=100, quantil_min_baseline=10,
                                       dview=dview)
Cn = Cns.max(axis=0)
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours(img=Cn)
plt.title('Contour plots of found components')

#%% save results
cnm.estimates.Cn = Cn
cnm.save(fname_new[:-5]+'_init.hdf5')

cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

# %% VIEW TRACES (accepted and rejected)
display_images = True
if display_images:
    cnm.estimates.view_components(images, img=Cn,
                                  idx=cnm.estimates.idx_components)
    cnm.estimates.view_components(images, img=Cn,
                                  idx=cnm.estimates.idx_components_bad)
#%% select neurons
good_discard = [28,41,53,55,64,91]
keep_neuron = [i for i in range(cnm.estimates.A.shape[1]) if i not in good_discard]
#bad_keep = []
#keep_neuron = [index for i,index in enumerate(cnm.estimates.idx_components_bad) if i in bad_keep] + [index for i,index in enumerate(cnm.estimates.idx_components) if i not in good_discard] 
print('original number of neurons:',cnm.estimates.A.shape[1],';Only {} neurons has left after selection'.format(len(keep_neuron)))
    
#%% update object with selected components
cnm.estimates.select_components(idx_components=keep_neuron)
print('All bad components has been removed; {} good neurons left.'.format(cnm.estimates.A.shape[1]))
#%% segmentation func
def slice_signal_to_trial(temp_traces,df): 
    # signal matrix: neurons * temporal traces; frames: the colomn for frame length, df.frame
    F_dff_individualtrials = {}
    cum_frames = np.cumsum(df.frames.values)
    cum_frames = np.insert(cum_frames, 0,0)
    for i in range(len(cum_frames)-1):
        F_dff_individualtrials['Trial'+str(i)] = temp_traces[:,cum_frames[i]:cum_frames[i+1]].copy()
        
    return F_dff_individualtrials

#%% Extract DF/F values and segment trials
from caiman.source_extraction.cnmf.utilities import detrend_df_f
dff_signals = detrend_df_f(cnm.estimates.A, cnm.estimates.b, cnm.estimates.C, cnm.estimates.f,
                             cnm.estimates.YrA*0.08, quantileMin=8, frames_window=1000, flag_auto=True, 
                             use_fast=False, detrend_only=False)

denoised_dff_segmented= slice_signal_to_trial(dff_signals,df) #能存到cnm哪里呢？
#%% Show final traces
cnm.estimates.view_components(img=Cn)
#%%
cnm.estimates.Cn = Cn
cnm.save(cnm.mmap_file[:-5] + '_processed.hdf5')
#%% reconstruct denoised movie (press q to exit)
if display_images:
    cnm.estimates.play_movie(images, q_max=99.9, gain_res=2,
                              magnification=2,
                              bpx=border_to_0,
                              include_bck=False)  # background not shown

#%% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)









































































































