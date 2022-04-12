# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:32:16 2021

@author: lechen qian
"""

#%% load packages and functions
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
from pathlib import Path
import pickle5
def pickle_dict(df,path,filename):
    try:
        os.makedirs(path) # create the path first
    except FileExistsError:
        print('the path exist.')
    filename = path +'/{}.pickle'.format(filename)
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

def getKeys(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
    return list

#%% load saved dicts
big_dict = load_pickleddata(
    'D:/PhD/Microscope/Big_dict_for_imaged_mice/5cond5deg_imaging_processed_partialaligned2021-07-14.pickle')
#%% mouse and corresponding activities
mouse_id = 'D2-02'
go_signal_matrix = big_dict[mouse_id]['aligned_dff_multitrials']['go']





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


































