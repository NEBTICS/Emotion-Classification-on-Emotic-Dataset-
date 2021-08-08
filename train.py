# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:18:24 2021

@author: smith
"""
import numpy as np
from tqdm import tqdm
import torch

class resnet_train():
    
    def train(train_dl,val_dl,net,device,optimizer,criterion):
        
        n_epochs = 1
        valid_loss_min = np.Inf
        val_loss = []
        val_acc = []
        train_loss = []  
        train_acc = []
        total_step = len(train_dl)
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            correct = 0
            total=0
            loop=tqdm(train_dl,leave=False,total=len(train_dl))
            for batch_idx, (data_, target_) in enumerate(loop):
                data_, target_ = data_.to(device), target_.to(device)
                optimizer.zero_grad()
                
                outputs = net(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                _,pred = torch.max(outputs, dim=1)
                total += target_.size(0)
        
                correct += torch.sum(pred==target_).item()
               
                loop.set_description(f'Epoch[{epoch}/{n_epochs}]')
                loop.set_postfix(Train_acc=correct/total,Train_loss=loss.item())
        
               
        
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss/total_step)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
            batch_loss = 0
            total_t=0
            correct_t=0
            with torch.no_grad():
                net.eval()
                bar=tqdm(val_dl,leave=False,total=len(val_dl))
                for data_t, target_t in bar:
                    data_t, target_t = data_t.to(device), target_t.to(device)
                    outputs_t = net(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _,pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t/total_t)
                val_loss.append(batch_loss/len(val_dl))
                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')
        
                
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(net.state_dict(), 'resnet.pt')
                    print('Improvement-Detected, save-model')
            net.train()