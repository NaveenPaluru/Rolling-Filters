#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:25:25 2020

@author: cds
"""


import os,time
import sklearn.metrics as metrics
import scipy.io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import torch.nn as nn
from datetime import datetime
from config import Config
from QSMnetDataset import mydataloader
#from anam3D import AnamNet
import torch.optim as optim
from torch.optim import lr_scheduler
from loss import*
from torch.utils.data import Dataset, DataLoader
from utils import*
from anam3D_perturb import AnamNet

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def test(directory):
    
    tic()
    net = AnamNet()
    net.load_state_dict(torch.load(directory))  
    
    
    config  = Config()   
     
   
    # make the data iterator for testing data . 
    testdata    = mydataloader('./test.csv', './Data/Testing Data/phs/', training = False)
    testloader  = DataLoader(testdata, batch_size = 1, shuffle=False, num_workers=1)
    
           
    if config.gpu == True:
        net = net.cuda(config.gpuid).eval()   
        par = torch.nn.DataParallel(net, device_ids=[0, 1])
    # define the train data stats

    stats = scipy.io.loadmat('./Data/Training Data/tr-stats.mat')
    
    b_mean= torch.tensor(stats['inp_mean'])
    b_std = torch.tensor(stats['inp_std' ])
    y_mean= torch.tensor(stats['out_mean'])
    y_std = torch.tensor(stats['out_std' ])  
    
    # directory to save the predictions
    out_data = './Data/Testing Data/anam_minus_15/'
    
    dw = -15/100
    for i,data in tqdm.tqdm(enumerate(testloader)):  
        
        # start iterations
        phs, N_dif, p_id = Variable(data[0]), data[1], data[2]
        
        idx = p_id[0].data.numpy()
        print('\n Generating Susceptibility Map of the Patient-', idx)
        #print(phs.shape)
        phs = (phs-b_mean)/b_std
        #print(phs.shape)
        # ckeck if gpu is available
        if config.gpu == True:
            phs  = phs.cuda(config.gpuid)
           
        # make forward pass      
        output   = par(phs,dw).detach().cpu() * y_std + y_mean
        #print(output.shape)
        pred_sus = crop_data(output.numpy(), N_dif[0])
        
        
        mdic = {"susc" : pred_sus}
        filename = out_data + 'net-' + str(idx) + ".mat"
        #print(filename)
        scipy.io.savemat(filename, mdic)
    toc()
        
                                                     #  KD       BNCK
                                                     
#"17Jan_1148am_model/"+ "AnamKD_12_model.pth"           Y         Y

#"16Jan_0612am_model/"+ "AnamAbal_12_model.pth"         N         Y

#"28Jan_0251pm_model/"+ "AnamAbalbneck_8_model.pth"     N         N

#"29Jan_0958am_model/"+ "AnamAbalbneck_14_model.pth"    Y         N



 

#"23Jan_0446pm_model/"+ "AnamKD_9_model.pth"     mini

# 24Jan_0106am_model/"+ "AnamAbal_5_model.pth"   mini
           
if __name__ == '__main__':         
        
    saveDir='./savedModels/'         
    # if want to test on a specific model
    directory=saveDir+"17Jan_1148am_model/"+ "AnamKD_12_model.pth" 
    print('Loading the Model : ', directory)    
    test(directory)