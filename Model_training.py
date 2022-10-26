# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:45:35 2022

                                Training model section 
    Input:  prepared dataset of the three different dataset micro patches 
    The model trained ensemble ways, using ensemble dataset and model architecture  
    The Saved training model  is : EResNet_trained_model.pth

"""
import math
import time
import sys
import os
import random
import glob
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import PIL.ImageOps 
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from pylab import *
use_gpu = torch.cuda.is_available()
import torch.nn.init as init

use_gpu=use_gpu
DATA_DIR = "/root/data/eyob_data"
trn_dir = "/root/data/eyob_data/train_data"
tst_dir ="/root/data/eyob_data/Test_data"

sz = 64
batch_size = 64



tfms = transforms.Compose([
    transforms.Resize((sz, sz//2)),  # PIL Image
#     transforms.Grayscale(), 
    transforms.ToTensor(),        # Tensor
    transforms.Normalize([0.44 , 0.053, 0.062], [0.076, 0.079, 0.085])
])


train_ds = datasets.ImageFolder(trn_dir, transform=tfms)
test_ds = datasets.ImageFolder(tst_dir, transform=tfms)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=8)
inputs, targets = next(iter(train_dl))



def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        if in_channels != out_channels or down:
            shortcut = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),           
                nn.LeakyReLU(0.1, inplace=True),]
        else:
            shortcut = []
        if down:
            shortcut.append(nn.MaxPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),           
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),           
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if down:
            residual.append(nn.MaxPool2d(2))
        self.residual = nn.Sequential(*residual)
        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.ResBlock1 = ResBlock(64, 64, down=False)
        self.ResBlock2 = ResBlock(64, 128, down=True)
        self.ResBlock3 = ResBlock(128, 128, down=False)
        self.ResBlock4_1 = ResBlock(128, 256, down=True)
        self.ResBlock4_2 = ResBlock(256, 256, down=True)

        self.fc1 = nn.Linear(256*8*4*2, 2)

             
    def forward(self, x, y, z):
        outx = self.conv1_1(x)
        #print(outx.shape)
        outx = self.ResBlock1(outx)
        #print(outx.shape)
        outx = self.ResBlock2(outx)
        #print(outx.shape)

        outx = self.ResBlock3(outx)
        #print(outx.shape)

        outx = self.ResBlock4_1(outx)
        #print(outx.shape)

        outx = outx.view(outx.size(0), -1)
        
        outy = self.conv1_2(y)
        outy = self.ResBlock1(outy)
        outy = self.ResBlock2(outy)
        outy = self.ResBlock3(outy)


        outz = self.conv1_3(z)
        outz = self.ResBlock1(outz)
        outz = self.ResBlock2(outz)
        outz = self.ResBlock3(outz)
        # outz = self.conv2_3(outz)
        # outz = self.conv3_3(outz)
        
        oyz = torch.cat([outy, outz],1)
        oyz = self.ResBlock4_2(oyz)
        
        # oyz = self.conv5(oyz)
        oyz = oyz.view(oyz.size(0), -1)
        
        oo=torch.cat([outx,oyz],1)
    
        out = self.fc1(oo)
        
        return out
model=CNN()

if use_gpu:
    
    model = model.cuda()
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


a = time.time()
num_epochs = 30
loss_values=[]

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dl):
        

        #targets = targets.type(torch.float)
        inputs = to_var(inputs)
        #inputs2 = to_var(inputs2)
        #inputs3 = to_var(inputs3)
        targets = to_var(targets)
        
        inputs1=inputs[:,0,:,:]
        inputs1=inputs1.resize(inputs1.shape[0],1,64,32)
        inputs2=inputs[:,1,:,:]
        inputs2=inputs1.resize(inputs2.shape[0],1,64,32)
        inputs3=inputs[:,2,:,:]
        inputs3=inputs1.resize(inputs3.shape[0],1,64,32)
        
        
        # forwad pass
        
        outputs = model(inputs1,inputs2,inputs3)
        outputs=outputs.squeeze(1)
        # loss
        loss = criterion(outputs, targets)
        #loss += loss.item()
        running_loss =+ loss.item() * inputs.size(0)
        loss_values.append(running_loss / len(train_dl))

        # backward pass
        loss.backward()

        
        # update parameters
        optimizer.step()
        
        #zero gradent
        
        optimizer.zero_grad()
        if loss.item() == 0.0000:
            break
        
        # report


        if (i + 1) % 1 == 0:
        
            print('Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_ds) // batch_size, loss.item()))
          
b = time.time()
print('Total Time of Training {:.1000}s'.format(b - a))
plt.figure(figsize=(8, 4))
plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.title('Cross Entropy Loss');

def evaluate_model(model, dataloader):
    model.eval()  # for batch normalization layers
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = to_var(inputs, True), to_var(targets, True)
#         targets = to_var(targets)
        
        inputs1=inputs[:,0,:,:]
        inputs1=inputs1.resize(inputs1.shape[0],1,64,32)
        inputs2=inputs[:,1,:,:]
        inputs2=inputs1.resize(inputs2.shape[0],1,64,32)
        inputs3=inputs[:,2,:,:]
        inputs3=inputs1.resize(inputs3.shape[0],1,64,32)
        
        outputs = model(inputs1,inputs2,inputs3)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()
        
    zz=len(dataloader.dataset)
    
    print('accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset)))
    print('corrects: {:.2f}'.format(corrects))
    print('Toatal: {:.2f}'.format(zz))

evaluate_model(model, train_dl)
evaluate_model(model, test_dl)
torch.save(model.state_dict(), 'EResNet_trained_model.pth')            
   