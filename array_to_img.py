# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 12:27:43 2021

@author: smith
"""
import pandas as pd
import numpy as np
import os
import torch
#import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image as im
 
data_src = r'A:\DOWNLOADS\emotioc\emotic_pre'


#%%
# Change data_src variable as per your drive



# Load training preprocessed data
train_context = np.load(os.path.join(data_src,'train_context_arr.npy'))
train_body = np.load(os.path.join(data_src,'train_body_arr.npy'))
train_cat = np.load(os.path.join(data_src,'train_cat_arr.npy'))
train_cont = np.load(os.path.join(data_src,'train_cont_arr.npy'))

# Load validation preprocessed data 
val_context = np.load(os.path.join(data_src,'val_context_arr.npy'))
val_body = np.load(os.path.join(data_src,'val_body_arr.npy'))
val_cat = np.load(os.path.join(data_src,'val_cat_arr.npy'))
val_cont = np.load(os.path.join(data_src,'val_cont_arr.npy'))

# Load testing preprocessed data
test_context = np.load(os.path.join(data_src,'test_context_arr.npy'))
test_body = np.load(os.path.join(data_src,'test_body_arr.npy'))
test_cat = np.load(os.path.join(data_src,'test_cat_arr.npy'))
test_cont = np.load(os.path.join(data_src,'test_cont_arr.npy'))

# Categorical emotion classes
cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(cat):
  cat2ind[emotion] = idx
  ind2cat[idx] = emotion
#%%
print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)
print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)
print ('completed cell')
#%%
train_context=np.concatenate((train_context,val_context))

#%%


print(len(train_context))
#%%
'''Creating all the 26 folders'''

string=r'C:\Users\smith\Videos\Emotic\test/'
for i in range(0,26):
    g=cat[i]
    os.mkdir(string+g)
    
#%%
'''Putting images in their respective folders'''
string = r'C:\Users\smith\Videos\Emotic\test/'
z=os.listdir(r'C:\Users\smith\Videos\Emotic\test/')
index=0
for i in tqdm(range(len(test_context)),leave=False):
    

    u=test_cat[i]
    v=test_context[i]
    temp_data=[]
    #print(len(u),u.dtype,u.shape)
    #print(u)
    for i in range(26) :
        if u[i]==1:
            #print(cat[i])
            temp_data.append(cat[i])
            z[i]==cat[i]
            g=z[i]
            os.chdir(string+g) # changing current workign directory
            data = im.fromarray(v) #saving the image
            save='.png'
            name=cat[i]
            data.save(name+str(index)+save)
            #print(os.getcwd())
            
    index+=1
print("Done")