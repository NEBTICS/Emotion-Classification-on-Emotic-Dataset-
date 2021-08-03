# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:54:05 2021

@author: smithbarbose
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from PIL import Image as im
import time

#%%
# Change data_src variable as per your drive
data_src = r'A:\DOWNLOADS\emotic\a\emotic_pre'


# Load training preprocessed data
train_context = np.load(os.path.join(data_src,'pre','train_context_arr.npy'))
train_body = np.load(os.path.join(data_src,'pre','train_body_arr.npy'))
train_cat = np.load(os.path.join(data_src,'pre','train_cat_arr.npy'))
train_cont = np.load(os.path.join(data_src,'pre','train_cont_arr.npy'))

# Load validation preprocessed data 
val_context = np.load(os.path.join(data_src,'pre','val_context_arr.npy'))
val_body = np.load(os.path.join(data_src,'pre','val_body_arr.npy'))
val_cat = np.load(os.path.join(data_src,'pre','val_cat_arr.npy'))
val_cont = np.load(os.path.join(data_src,'pre','val_cont_arr.npy'))

# Load testing preprocessed data
test_context = np.load(os.path.join(data_src,'pre','test_context_arr.npy'))
test_body = np.load(os.path.join(data_src,'pre','test_body_arr.npy'))
test_cat = np.load(os.path.join(data_src,'pre','test_cat_arr.npy'))
test_cont = np.load(os.path.join(data_src,'pre','test_cont_arr.npy'))

# Categorical emotion classes
cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(cat):
  cat2ind[emotion] = idx
  ind2cat[idx] = emotion

print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)
print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)
print ('completed cell')
#%%
'''Creating all the 26 folders'''

string=r'A:\DOWNLOADS\emotic\a\Dataset/'
for i in range(0,26):
    g=cat[i]
    os.mkdir(string+g)
    
#%%
'''Putting images in their respective folders'''
string = r'A:\DOWNLOADS\emotic\a\Dataset/'
z=os.listdir(r'A:\DOWNLOADS\emotic\a\Dataset')
index=0
for i in range(500):
    
    
    u=val_cat[i]
    v=val_context[i]
    temp_data=[]
    print(len(u),u.dtype,u.shape)
    print(u)
    for i in range(26) :
        if u[i]==1:
            print(cat[i])
            temp_data.append(cat[i])
            z[i]==cat[i]
            g=z[i]
            os.chdir(string+g) # changing current workign directory
            data = im.fromarray(v) #saving the image
            save='.png'
            name=cat[i]
            data.save(name+str(index)+save)
            print(os.getcwd())
    index+=1
        
        
    



        

    
