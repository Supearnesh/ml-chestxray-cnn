import os
import math
import random as rn
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T, datasets

class Net(pl.LightningModule):

    def __init__(self, train_data_dir, batch_size=128, test_data_dir=None, num_workers=4):

            super(Net, self).__init__()

            self.batch_size = batch_size
            self.train_data_dir = train_data_dir
            self.test_data_dir = test_data_dir
            self.num_workers = num_workers

            ## Define layers of a CNN
            # convolutional layer
            self.conv_01 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=28, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            # convolutional layer
            self.conv_02 = nn.Sequential(
                nn.Conv2d(in_channels=28, out_channels=10, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            # add dropout layer
            self.dropout_01 = nn.Dropout(0.25)

            # linear layer
            self.fc_01 = nn.Linear(in_features=55180, out_features=4096)

            # add dropout layer
            self.dropout_02 = nn.Dropout(0.16)

            # linear layer
            self.fc_02 = nn.Linear(in_features=4096, out_features=2)

    def load_split_train_test(self, valid_size = .2):

        num_workers = self.num_workers

        train_transforms = T.Compose([T.Resize(256),
                                           T.Resize((256,364)),
                                           T.ToTensor(),
                                           T.Normalize(mean=(0.4953996,), std=(0.0565181,))])
        
        train_data = datasets.ImageFolder(self.train_data_dir,
                                          transform=train_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)        

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   sampler=train_sampler,
                                                   batch_size=self.batch_size,
                                                   num_workers=num_workers)
        
        val_loader = torch.utils.data.DataLoader(train_data,
                                                 sampler=val_sampler,
                                                 batch_size=self.batch_size,
                                                 num_workers=num_workers)

        test_loader = None
        
        if self.test_data_dir is not None:
            test_transforms = T.Compose([T.Resize(256),                                       
                                           T.CenterCrop(224),
                                           T.ToTensor(),
                                           T.Normalize(mean=(0.4953996,), std=(0.0565181,))])
            
            test_data = datasets.ImageFolder(self.test_data_dir,
                                             transform=test_transforms)
            
            test_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=self.batch_size,
                                                      num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def prepare_data(self):
        self.train_loader, self.val_loader, self.test_loader  = self.load_split_train_test()
        
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return DataLoader(ChestXRay(os.getcwd(), train=False, download=False, transform=transform.ToTensor()), batch_size=128)
    
    def forward(self,x):
        x = self.conv_01(x)
        x = self.conv_02(x)
        x = self.dropout_01(x)
        x = torch.relu(self.fc_01(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.dropout_02(x))
        
        return F.softmax(self.fc_02(x), dim=1)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, labels = batch
        prediction = self.forward(x)
        loss = F.nll_loss(prediction, labels)
        logs={'train_loss':loss}
        
        output = {
            'loss':loss,
            'log':logs
        }
        
        return output

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        prediction = self.forward(x)
        
        return {
            'val_loss': F.cross_entropy(prediction, labels)
        }
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        return {'val_loss': val_loss_mean}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Average training loss: '+str(avg_loss.item()))
        logs = {'val_loss':avg_loss}
        
        return {
            'avg_val_loss':avg_loss,
            'log':logs
        }