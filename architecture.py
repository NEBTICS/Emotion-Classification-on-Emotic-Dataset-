# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 20:01:53 2021

@author: smith
"""
import torch.nn as nn
import torchvision
from torchvision import *
from torchsummary import summary

class architecture():
    def resnet50(out_labels):
        net = models.resnet50(pretrained=True)
        print("model downloaded")
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_labels)
        print("Changes DONE")
        print(net)
        return net
    
    def load_past_model(model,path):
        try:
            model.load_state_dict(torch.load(path))
            return model
            print("Past model loaded")
        except FileNotFoundError:
            print('Invalid file path---> if you are training this model for the first time skip this else check the save path with extension as resnet.pt ---Regards Barbose')
        
    def summary(model,RGB,H,W):       
            summary(model,(R,H,W))

