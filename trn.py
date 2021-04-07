#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:25:00 2020

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
from myDataset import mydataloader
from student import EQSMnet
from teacher import QSMnet
import torch.optim as optim
from torch.optim import lr_scheduler
from loss import*
from torch.utils.data import Dataset, DataLoader


print ('*******************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model'
print('Model will be saved to  :', directory)

if not os.path.exists(directory):
    os.makedirs(directory)

config  = Config()

traindata   = mydataloader('./train.csv', './Data/Training Data/')
trainloader = DataLoader(traindata, batch_size = config.batchsize, shuffle=True, num_workers=1)

valdata    = mydataloader('./val.csv', './Data/Validation Data/')
valloader  = DataLoader(valdata, batch_size = config.batchsize, shuffle=True, num_workers=1)

print('----------------------------------------------------------')
#%%
# Create the object for the network

if config.gpu == True:    
    net = EQSMnet().cuda(config.gpuid)
    par = torch.nn.DataParallel(net, device_ids=[0, 1])
    tchr= QSMnet()
    tchr.load_state_dict(torch.load(saveDir+"27Dec_0244pm_model/"+ "QSMnet_25_model.pth"))
    for p in (tchr.parameters()):
        p.requires_grad = False
    tchr=tchr.cuda(config.gpuid).eval()
    par2 = torch.nn.DataParallel(tchr, device_ids=[0, 1])

   
# Define the optimizer
optimizer = optim.Adam(net.parameters(),lr=0.005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# define dipole kernel (to be used in loss function)

matrix_size = [64, 64, 64]
voxel_size  = [1,  1,  1 ]
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0,0,1])
dk = dk.repeat(config.batchsize,1,1,1,1)
dk = dk.cuda(config.gpuid)

# define sobel kernel (to be used in loss function)

ss = sobel_kernel()
ss = ss.cuda(config.gpuid)

# define the train data stats

stats = scipy.io.loadmat('./Data/Training Data/tr-stats.mat')

b_mean= torch.tensor(stats['inp_mean'])
b_std = torch.tensor(stats['inp_std' ])
y_mean= torch.tensor(stats['out_mean'])
y_std = torch.tensor(stats['out_std' ])


# Iterate over the training dataset
train_loss = []
val_loss = []

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
tic()
for j in range(config.epochs):  
    # Start epochs   
    runtrainloss = 0    
    net.train() 
    for i,data in tqdm.tqdm(enumerate(trainloader)): 
        
        # start iterations
        b, m, y = Variable(data[0]), Variable(data[1]), Variable(data[2])         
        # y:cosmos, b:phs and m:mask
        
        # ckeck if gpu is available
        if config.gpu == True:
            b  = b.cuda(config.gpuid )
            m  = m.cuda(config.gpuid )
            y  = y.cuda(config.gpuid )
            b_mean= b_mean.cuda(config.gpuid)
            b_std = b_std.cuda( config.gpuid)
            y_mean= y_mean.cuda(config.gpuid)
            y_std = y_std.cuda( config.gpuid)
            b = (b-b_mean)/b_std
            y = (y-y_mean)/y_std
                                
        # make forward pass  for student     
        chi = par(b)
        
        # make forward pass  for teacher     
        t   = par2(b)
       
        #compute loss with GT
        loss1   = total_loss(chi, y, b, dk, m, b_mean, b_std, y_mean, y_std, ss)  
        
        #compute loss with teacher
        loss2   = total_loss(chi, t, b, dk, m, b_mean, b_std, y_mean, y_std, ss)  
        
        # compute loss
        alpha = 0.7
        loss  = alpha * loss1 + (1-alpha) * loss2    
                
        # make gradients zero
        optimizer.zero_grad()
        
        # back propagate
        loss.backward()
        
        # Check for back propability of losses
        # print(net.conv1_block[0].weight.grad.data)
        
        # Accumulate loss for current minibatch
        runtrainloss += loss.item()        
        
        # update the parameters
        optimizer.step()       
        
     
    # print train loss
    
    print('\n Training - Epoch {}/{}, loss:{:.9f} '.format(j+1, config.epochs, runtrainloss/len(trainloader)))    
    train_loss.append(runtrainloss/len(trainloader))
    
    
    # start validation after each epoch
    
    runvalloss = 0
    net.eval()  

    for i,data in tqdm.tqdm(enumerate(valloader)):         
      
        # start iterations
        b, m, y = Variable(data[0]), Variable(data[1]), Variable(data[2])         
        # y:cosmos, b:phs and m:mask
        
        # ckeck if gpu is available
        if config.gpu == True:
            b  = b.cuda(config.gpuid )
            m  = m.cuda(config.gpuid )
            y  = y.cuda(config.gpuid )
            b_mean= b_mean.cuda(config.gpuid)
            b_std = b_std.cuda( config.gpuid)
            y_mean= y_mean.cuda(config.gpuid)
            y_std = y_std.cuda( config.gpuid)
            b = (b-b_mean)/b_std
            y = (y-y_mean)/y_std
                                
        # make forward pass  for student     
        chi = par(b)
        
        # make forward pass  for teacher     
        t   = par2(b)
       
        #compute loss with GT
        loss1   = total_loss(chi, y, b, dk, m, b_mean, b_std, y_mean, y_std, ss)  
        
        #compute loss with teacher
        loss2   = total_loss(chi, t, b, dk, m, b_mean, b_std, y_mean, y_std, ss)  
        
        # compute loss
        alpha = 0.7
        loss  = alpha * loss1 + (1-alpha) * loss2    
       
          
        # Accumulate loss for current minibatch
        runvalloss += loss.item()
        
          
    # print val loss    
    
    print('\n Validatn - Epoch {}/{}, loss:{:.9f} '.format(j+1, config.epochs, runvalloss/len(valloader)))
    print('----------------------------------------------------------')
    val_loss.append(runvalloss/len(valloader))  
    
    
    # Take a step for scheduler
    scheduler.step()    
    
    #save the model   
    torch.save(net.state_dict(),os.path.join(directory,"EQSMnet_" + str(j+1) +"_model.pth"))
            

# Save the train stats

np.save(directory+'/trnloss.npy',np.array(train_loss) )
np.save(directory+'/valloss.npy',np.array(val_loss) )

toc()
# plot the training loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.plot(x,val_loss,label='Validation')
plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper left")  
plt.show()


                      



   




