#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:19:54 2020

@author: cds
"""


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

import numpy as np
import torchvision.transforms as transforms
import scipy.io
from torch.autograd import Variable
from utils import*

class mydataloader(Dataset):
    
    def __init__(self, csv_file, root_dir, training = True):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.training = training
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        if self.training==True:
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])             
            data = scipy.io.loadmat(file_name)
            phs  = torch.tensor(data['phs' ]).unsqueeze(dim=0)
            msk  = torch.tensor(data['msk' ]).unsqueeze(dim=0)
            sus  = torch.tensor(data['susc']).unsqueeze(dim=0)
            return phs, msk, sus
        else:
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])  
            p_id = self.names['Label'][idx]
            data = scipy.io.loadmat(file_name)
            #print(data['phs'].shape)
            phs, N_dif, N_16 = padding_data(data['phs'])
            #print(phs.shape)
            phs  = torch.tensor(phs).squeeze(dim=0)
            return phs, N_dif, p_id
            


## Check the dataloader

if __name__=="__main__":
    
    loader = mydataloader('./train.csv', './Data/Training Data/')
    trainloader = DataLoader(loader, batch_size = 1, shuffle=False, num_workers=1)
    print(len(trainloader))
    for i, data in enumerate(trainloader): 
        phs, msk, sus = Variable(data[0]), Variable(data[1]), Variable(data[2])
        print(i)
        print(phs.size())
        print(msk.size())
        print(sus.size())
        print('-------------------------------------------------------------------');
        