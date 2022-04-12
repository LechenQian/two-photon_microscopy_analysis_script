# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:03:34 2020

@author: qianl
"""
#%% import packages
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

#%% define functions
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
#%% read files
#################### change
filedir = 'D:/PhD/Microscope/Selina/imaging_data/imaging_data_2020_3_Pav/OT-GC-3/OT-GC-3_2020-03-15-session2_deg' # maybe can be changed to input
session = 7
sep_id_str = input('Please specify the separate session for conditioning and degradation:')
sep_id = int(sep_id_str)

file_index, file_frame, filenames = read_image_file(filedir)
index = pd.Series(file_index, name = 'file_index')
frame = pd.Series(file_frame, name = 'frames')
file_name = pd.Series(filenames, name = 'filename')
df = pd.concat([index,frame, file_name], axis=1)


#%% import behavioral data
#################### change
sys.path.insert(0,'D:/PhD/Behavior/behavior_20_03/functions')
from parse_data import Mouse_200310
#################### change
mouse_id = 'OT-GC-3'
load_path = 'D:/PhD/Behavior/behavior_20_03/parsed_dataframe_pickle/{0}_stats.pickle'.format(mouse_id)
mouse = load_pickleddata(load_path)
#%% compare solenoid signal with behavioral record
data = np.fromfile(filedir+'/OT-GC-3T95.data', dtype='>f8')
plt.plot(data)
# mouse.df_trials[mouse.all_days[session]]

#%% check and delete trials if doesn't match
# When you delete a row now in the dataframe here, deleting the corresponding files in the folder!!! if you delete one later when you are selecting trials, don't delete that from the imaging dataset.
df.drop(index = df.index[[0]], inplace = True)
df.reset_index(drop=True,inplace = True)
filenames = df.filename.values

#%% create folders for average and processed images of each session
ans = input('Do you want to create folders for processed images?(y or n')
if ans == 'y':
    # create folders for OT-GC-1 PROCESSED DOLFER
    
    for i in range(len(mouse.all_days)):        
        path = 'D:/PhD/Microscope/Selina/imaging_data/processed/{2}/000{0}_{1}'.format(i, 'cond' if i <= sep_id else 'deg',mouse_id )
        try:
            os.makedirs(path) # create the path first
        except FileExistsError:
            print('the path exist.')

#%% MOTION CORRECTION I
# preparation part
# filepath with the bracket
# filenames = ['D:/PhD/Microscope/Selina/imaging_data/processed/OT-GC-1/0009_03-16-14-00_deg/0009_tiff_chained_all_corrected.tiff']
tiff_chained = cm.load_movie_chain(filenames)
print(tiff_chained.shape)  

# save chained tiffs
is_save = int(input('save chained tiffs? (1 or 0)'))
savedir = 'D:/PhD/Microscope/Selina/imaging_data/processed/{0}/000{1}_{3}'.format(mouse_id,session, 'cond' if session <= sep_id else 'deg')
stack_name = [savedir+'/000{}_tiff_chained_all.tiff'.format(session)]

if is_save:
    tif.imsave(stack_name[0], tiff_chained.astype('uint16'), bigtiff=True)          

#Visualize tiff_chained images
tiff_chained.play(magnification = 2, fr = 60, q_min = 0.1, q_max = 99.75)
  
#%% MOTION CORRECTION II
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
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=8, single_thread=False)

# motion correction
mc = MotionCorrect(stack_name, dview=dview, **opts.get_group('motion')) #object

# Run piecewise-rigid motion correction using NoRMCorre
mc.motion_correct(save_movie=True)

m_els = cm.load(mc.fname_tot_els)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 
mmap_name= mc.mmap_file[0]
print(mmap_name)

# things will be saved here: 1. motion-corrected tiff file

save_correctedimage = True
if save_correctedimage:
    corrected_name = [stack_name[0][0:-5]+'_corrected.tiff']
    tif.imsave(corrected_name[0], m_els.astype('float32'), bigtiff=True)

#%% MEMORY MAPPING
    
# things will be saved here: 1. mmap images 2. correlogram
    
# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview=dview)
Yr, dims, T = cm.load_memmap(fname_new)

# this images will be used for CNMF algorithm
images = np.reshape(Yr.T, [T] + list(dims), order='F') #load frames in python format (T x X x Y)
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
tif.imsave(savedir+'/correlogram.tiff', Cn.astype('float32'), bigtiff=True)

#%% restart cluster to clean up memory

cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=8, single_thread=False)

# things will be saved here: 1. average motion-corrected image
plt.figure(figsize = (10,10))
io.imshow(images.mean(axis = 0))
plt.savefig(savedir+'/mean.png', bbox_inches="tight", dpi = 400)

#%% compare raw movie and motion-corrected movie

display_movie = True # False to skip
if display_movie:
    m_orig = tiff_chained.copy()
    ds_ratio = 1
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                    m_els.resize(1, 1, ds_ratio)], 
                   axis=2).play(fr=30, gain=1, magnification=2, offset=0)  # press q to exit




































































































































































































































