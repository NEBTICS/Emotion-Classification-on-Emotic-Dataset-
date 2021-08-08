# -*- coding: utf-8 -*-
"""
@author: smith
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import time
import copy
import os

data_dir = r'C:\Users\smith\Videos\Emotic'
#%%
print(os.listdir(data_dir)) #folders in the dataset folder
classes = os.listdir(data_dir + "/train")
classes_list=classes
print(classes)

#%%
'''Data transforms (Gray scaling and data augmentation'''
mean = [0.4690646, 0.4407227, 0.40508908]
std = [0.2514227, 0.24312855, 0.24266963]


train_trams=tt.Compose([tt.Resize((224,224)),
                        tt.RandomHorizontalFlip(),tt.ToTensor(),tt.Normalize(mean,std,inplace=True)])
#test
test_trams=tt.Compose([tt.Resize((224,224)),tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])


#%%
''' datasets loading'''
train_ds=ImageFolder(data_dir+'/train',train_trams)
val_ds=ImageFolder(data_dir+'/test',test_trams)


#%%
batch_size=50
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
    

#%%
import GPU 
from GPU import gpu
device = gpu.get_default_device()
print(device)
#%%
'''Architecture '''
# Specify a path
import architecture
from architecture import architecture

PATH = "A:/BE/emotion/resnet.pt"
model=architecture.resnet50(26)
#%%
model=architecture.load_past_model(model,PATH)


#%%
train_dl=gpu.DeviceDataLoader(train_dl, device)
val_dl=gpu.DeviceDataLoader(val_dl, device)
gpu.to_device(model, device)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


#%%
import train
from train import resnet_train

resnet_train.train(train_dl,val_dl,model,device,optimizer,criterion)





#%%
from torchsummary import summary

summary(net,(3,224,224))