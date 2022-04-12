# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:12:37 2021

@author: qianl
"""
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
    
    
def find_file_from_dir(filedir,include_str=None,exclude_str=None):

    filepaths = []
    count = 0
    # can walk through all levels down
    for dirpath, dirnames, files in os.walk(filedir):
        #     print(f'Found directory: {dirpath}')
        for f_name in files:
            if any([True for token in include_str if token in f_name]):
                filename = dirpath+'/'+f_name
                filepaths.append(filename)
                count +=1
    print(dirpath,'found {}'.format(count))


    return filepaths

def read_excels_into_dict(path_list,keys_list):
    reg_dict = {}
    if len(path_list) != len(keys_list):
        raise ValueError('The length of path list and key list arent the same')
    else:
        for key,path in zip(keys_list,path_list):
            reg_data = pd.read_excel(path, header=None)
            reg_data.replace('Nan',np.nan,inplace = True)

            assignments = reg_data.values
            
            order = np.argsort(np.sum(np.isnan(assignments),axis = 1))
            assignments_new = assignments[order,:]
            reg_dict[key] = assignments_new
           
    return reg_dict
        
def getKeys(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list
#%%


master_dir = 'D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled/registration'
mouse_id = os.listdir(master_dir)
excel_paths = []
for mouse in mouse_id:
    filedir = os.path.join(master_dir,mouse)
    excel_path_list = find_file_from_dir(filedir,include_str=['registration'],exclude_str=None)
    excel_paths += excel_path_list
    
assignment_dict = read_excels_into_dict(excel_paths,mouse_id)

#%% load the assignment dictionary into big_dict
filename = "D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled/from_mice_pickle_and_hdf5_loaded_processed_2021-10-13.pickle"
big_dict = load_pickleddata(filename)
print(getKeys(big_dict))
#%%
for key, value in assignment_dict.items():
    big_dict[key]['registration_mat'] = value
print(getKeys(big_dict['D1-02']))
print(getKeys(big_dict['D1-01']))

path = "D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled"
filename = "from_mice_pickle_and_hdf5_loaded_processed_2021-10-13_registration_added.pickle"
pickle_dict(big_dict, path, filename)



































































































































































































