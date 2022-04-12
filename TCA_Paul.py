# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:09:46 2021

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
import tensortools as tt
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
    if partial:
        index_list = input('enter a list of indexes you want to concat:>>  ')
        index_list = json.loads(index_list)
    for mouse in mice_name:
        if not partial:
            temp_list = [item for item in data[mouse][keyname][trialtype]]
            stacked_multisesions_traces = np.dstack(temp_list)
        else:
            
            temp_list = [data[mouse][keyname][trialtype][i] for i in index_list]
            try:
                stacked_multisesions_traces = np.dstack(temp_list)
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


filename ="D:/PhD/Microscope/Big_dict_for_imaged_mice/all_pickled/from_mice_pickle_and_hdf5_loaded_processed_2021-10-13.pickle"
partial_dict_tomac = load_pickleddata(filename)
#%%
deg_d1 = ['D1-02','D1-05','D1-12','D1-13']
deg_d2 = ['D2-02','D2-04','D2-16','D2-17']
con_d1 = ['D1-15']
con_d2 = ['D2-18','D2-21','D2-23','D2-24']

deg_d1_tmat = stack_traces_from_multi_mice(deg_d1, partial_dict_tomac, 'registered_aligned_dff_average', 'go')
deg_d2_tmat = stack_traces_from_multi_mice(deg_d2, partial_dict_tomac, 'registered_aligned_dff_average', 'go')
con_d1_tmat = stack_traces_from_multi_mice(con_d1, partial_dict_tomac, 'registered_aligned_dff_average', 'go')
con_d2_tmat = stack_traces_from_multi_mice(con_d2, partial_dict_tomac, 'registered_aligned_dff_average', 'go')
#%%%

data = con_d1_tmat# ... specify a numpy array holding the tensor you wish to fit
data = np.nan_to_num(data)
# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
ensemble = tt.Ensemble(fit_method="cp_als")
ensemble.fit(data, ranks=range(1, 10), replicates=4)

fig, axes = plt.subplots(1, 2)
tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
fig.tight_layout()
#%%
# Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
num_components = 6

replicate = 0
tt.plot_factors(ensemble.factors(num_components)[replicate])  # plot the low-d factors

plt.show()
#%%
neuron_weights_mat = np.zeros([data.shape[0],num_components])
U = ensemble.factors(num_components)[replicate]
for r in range(U.rank):
    
    for i, f in enumerate(U):
        if i == 0:
            neuron_weights_mat[:,r] = f[:, r]



# ax = sns.heatmap(neuron_weights_mat, linewidth=0.5)

sns.clustermap(neuron_weights_mat,cmap="vlag",vmin = -1,vmax = 1)
plt.show()

#%%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

 
#Load Data

pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(neuron_weights_mat)
print(df.shape)
plt.scatter(df[:,0],df[:,1])
#%%

 from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)

















