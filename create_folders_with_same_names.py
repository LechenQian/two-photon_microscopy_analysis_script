# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:34:09 2022

@author: qianl
"""
import os
#%% read folder names from a directory

mouse_id = 'D2-24'
group = 'control'
month = '2021_09'
directory = "D:/PhD/Microscope/current_analysis/imaging_data_{}_Pav/{}/processed".format(month,mouse_id)
subdirs = [ x[1] for x in os.walk(directory) ][0]
print(subdirs)


#% create folders in a specified directory

for subdir in subdirs:
    path = os.path.join('D:/PhD/Microscope/Big_dict_for_imaged_mice/reanalyzed_10session_pickle_and_hdf5/{}/{}/{}/sup'.format(mouse_id[0:2],group,mouse_id),subdir)
    os.makedirs(path)