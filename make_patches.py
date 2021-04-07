#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 14:33:02 2020

@author: cds
"""

"""
This snippet has been taken from https://github.com/SNU-LIST/QSMnet
"""
import scipy.io
import numpy as np
import h5py
import time
import os
import sys

"""
The patches will be saved as : 
    tr-patient-orientation-patch_number
"""

'''
File Path
'''

data_path = './Raw Data/TR-'     # TR- Training Data, # VAL - Validation Data
out_data = './Data/Training Data/'

start_time = time.time()

'''
Constant Variables
'''
PS = 64  # Patch size
sub_num = 10  # number of subjects (= 1 for Validation)
dir_num = 1  # number of directions (I am loading single orientation)
patch_num = [6, 8, 7]  # Order of Dimensions: [x, y, z]

'''
Code Start
'''

# Patch the input & mask file ----------------------------------------------------------------


print("####patching input####")

patches_field = []
patches_mask  = []
patches_susc  = []


for dataset_num in range(1, sub_num + 1):      
    
    for orients in range(1, 6):   
        
        patch_number = 1
        
        field = scipy.io.loadmat(data_path + str(dataset_num) + '/phs' + str (orients) +'.mat')
        mask  = scipy.io.loadmat(data_path + str(dataset_num) + '/msk' + str (orients) +'.mat')
        susc  = scipy.io.loadmat(data_path + str(dataset_num) + '/cos' + str (orients) +'.mat')
        matrix_size = np.shape(field['phs'])
        strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
        
        if np.size(matrix_size) == 3:
            field['phs'] = np.expand_dims(field['phs'], axis = 3)
            susc['cos']  = np.expand_dims(susc['cos'],  axis = 3)           
            mask['msk']  = np.expand_dims(mask['msk'],  axis = 3)
            matrix_size = np.append(matrix_size, [1],   axis = 0)
        
        if matrix_size[3] < dir_num:
            sys.exit("dir_num is bigger than data size!")
        
        for idx in range(dir_num):
            
            for i in range(patch_num[0]):
                
                for j in range(patch_num[1]):
                    
                    for k in range(patch_num[2]):
                        
                        phs_tmp = field['phs'][  i * strides[0]:i * strides[0] + PS,
                                                 j * strides[1]:j * strides[1] + PS,
                                                 k * strides[2]:k * strides[2] + PS, idx]
                        patches_field.append(phs_tmp)
                        
                        msk_tmp = mask['msk'][   i * strides[0]:i * strides[0] + PS,
                                                 j * strides[1]:j * strides[1] + PS,
                                                 k * strides[2]:k * strides[2] + PS, idx]                        
                        patches_mask.append(msk_tmp)
                        
                        susc_tmp = susc['cos'][  i * strides[0]:i * strides[0] + PS,
                                                 j * strides[1]:j * strides[1] + PS,
                                                 k * strides[2]:k * strides[2] + PS, idx]                        
                        
                        patches_susc.append(susc_tmp)
                        
                        mdic = {"phs": phs_tmp, "msk": msk_tmp, "susc":susc_tmp}
                        filename = out_data+"tr-" + str(dataset_num) + '-' + str(orients) + '-' + str(patch_number) + ".mat"
                        #print(filename)
                        scipy.io.savemat(filename, mdic)
                        patch_number=patch_number+1
                        
print("Done!")

patches_field = np.array(patches_field, dtype='float32', copy=False)
patches_mask  = np.array(patches_mask,  dtype='float32', copy=False)
patches_susc  = np.array(patches_susc,  dtype='float32', copy=False)

patches_field = np.expand_dims(patches_field,axis=4)
patches_mask  = np.expand_dims(patches_mask, axis=4)
patches_susc  = np.expand_dims(patches_susc, axis=4)

print("Total input  data size : " + str(np.shape(patches_field)))
print("Total output data size : " + str(np.shape(patches_field)))

input_mean = np.mean(patches_field[patches_mask > 0])
input_std  = np.std (patches_field[patches_mask > 0])

output_mean = np.mean(patches_susc[patches_mask > 0])
output_std  = np.std (patches_susc[patches_mask > 0])

mdic2 = {"inp_mean": input_mean, "inp_std": input_std, "out_mean":output_mean, "out_std": output_std}
filename2 = out_data+"tr-stats" + ".mat"
scipy.io.savemat(filename2, mdic2)




