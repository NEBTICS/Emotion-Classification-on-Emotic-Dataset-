# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:58:15 2021
I have use UCF101 dataset to test the model

@author: smith
"""
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
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = r'C:\Users\smith\Videos\UCF101\AutoEncoder-Video-Classification-master'
#%%
print(os.listdir(data_dir)) #folders in the dataset folder
classes = os.listdir(data_dir + "/test")
classes_list=classes
print(classes)

#%%
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
img , label= dataset[9000]
print(f'{img.shape} label== {label}')
#print(img)
#%%
'''Ploting a single image'''
img,dataset.classes[label]

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    
show_example(*dataset[10000])


#%%
'''Data transforms (Gray scaling and data augmentation'''
train_trams=tt.Compose([tt.Resize((100,100)),
                        tt.RandomHorizontalFlip(),tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])
#test
test_trams=tt.Compose([tt.Resize((100,100)),tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])
#%%
''' datasets loading'''
train_ds=ImageFolder(data_dir+'/train',train_trams)
val_ds=ImageFolder(data_dir+'/test',test_trams)

#%%
for i in range(50):
    image,l=train_ds[i]
    print(image.shape)

#%%
batch_size=10
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
#%%

#%%
'''Ploting bunch of images'''
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    
show_batch(train_dl)
#%%
'''Defining model'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        print('Validation accuracy: %d %%' % (100*epoch_acc.item()))
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Fruits360CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 50 x 50

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 25 x 25

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),#output :256*25*25
            nn.MaxPool2d(5, 5), # output: 256 x 5 x 5

            nn.Flatten(), 
            nn.Linear(256*5*5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 101))
            
        
    def forward(self, xb):
        return self.network(xb)
#%%
model = Fruits360CnnModel()
print(model)

#%%
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
print(device)
#%%
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)
#%%
'''Traning'''

@torch.no_grad()#stop SGD
def evaluate(model,val_loader):
    model.eval()
    bar=tqdm(val_loader,leave=False,total=len(val_loader))
    outputs=[model.validation_step(batch) for batch in bar]
    print("Model Validation--",model.validation_epoch_end(outputs))
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    train_accuracy_plot=[]
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        loop=tqdm(train_dl,leave=False,total=len(train_dl))
        # Training Phase 
        correct=0
        total=0
        model.train()
        train_losses = []
        for batch in loop:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #traning acc
            images, labels = batch
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()            
            
            
            
            #updating the bar
            loop.set_description(f'Epoch[{epoch}/{num_epochs}]')
            loop.set_postfix(Train_acc=correct/total,Train_loss=loss.item())
              
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        print('Traning accuracy: %d %%' % (100*correct / total))
        model.epoch_end(epoch, result)
        

        history.append(result)
        train_accuracy_plot.append(correct/total)    
    return history,train_accuracy_plot

#%%
model = to_device(Fruits360CnnModel(), device)
#%%
evaluate(model, val_dl)

#%%
num_epochs = 2
opt_func = torch.optim.Adam
lr = 0.001
#%%
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)



#%%
train_accu= history[1]
eval_accu = [x['val_acc'] for x in history[0]]
plt.plot(train_accu,'-o')
plt.plot(eval_accu,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')
 
plt.show()


#%%
torch.save(model.state_dict(), 'fruits-cnn.pth')

#%%
nb_classes = 6

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_dl):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
#%%

    

#%%

#%%
import pandas as pd
import seaborn as sns

nb_classes = 6
confusion_matrix = torch.zeros((nb_classes, nb_classes))
with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_dl):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))

class_names = classes_list
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')

#%%
a=confusion_matrix.diag()/confusion_matrix.sum(1)
a=a.tolist()
for i in range(6):
    print(f'{classes[i]} = {a[i]}')


#%%
model1=torch.load('A:/DOWNLOADS/dataset/fruits-cnn.pth')











