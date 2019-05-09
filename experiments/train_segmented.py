from __future__ import division
import time
import os
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-model_file', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-dataset', type=str, default='i3d')

args = parser.parse_args()
print args

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
import json

import models
import segmented_dataset as sd
from apmeter import APMeter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



batch_size = 32
dataset = sd.SegmentedPitchResultMultiLabel('mlb-youtube-segmented.json', 'mlb-youtube-negative.json', 'training', '/')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sd.collate_fn)

val_dataset = sd.SegmentedPitchResultMultiLabel('mlb-youtube-segmented.json', 'mlb-youtube-negative.json', 'testing', '/')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=sd.collate_fn)

dataloaders = {'train': dataloader, 'val': val_dataloader}
datasets = {'train': dataset, 'val': val_dataset}

def pool(f, s, e):
    s = int(s)
    e = int(e)
    return torch.max(f[:,:, s:e], dim=2)[0].unsqueeze(2)

# train the model
def train_model(model, criterion, optimizer, num_epochs=50):
    since = time.time()
    val_res = {}
    best_acc = 0

    for epoch in range(num_epochs):
        print 'Epoch {}/{}'.format(epoch, num_epochs - 1)
        print '-' * 10

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            apm = APMeter()
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            error = 0.0
            num_iter = 0.

            tr = {}
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                features, mask, labels, name = data


                # wrap them in Variable
                features = Variable(features.cuda())
                labels = Variable(labels.float().cuda())
                mask = Variable(mask.cuda())#.unsqueeze(1)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                                
                # forward

                #un-comment for max-pooling
                #features = torch.max(features, dim=2)[0].unsqueeze(2)
                #outputs = model(features)
                
                # un-comment for pyramid
                #b,c,t,h,w = features.size()
                #features = [pool(features,0,t), pool(features,0,t/2),pool(features,t/2,t),pool(features,0,t/4),pool(features,t/4,t/2),pool(features,t/2,3*t/4),pool(features,3*t/4,t)]
                #features = torch.cat(features, dim=1)
                #outputs = model(features)


                # sub-event learning
                outputs = model([features, torch.sum(mask, dim=1)])


                outputs = outputs.squeeze() # remove spatial dims
                if features.size(0) == 1:
                    outputs = outputs.unsqueeze(0)
                #outputs = outputs.permute(0,2,1)

                # action-prediction loss
                loss = criterion(outputs, labels)

                probs = torch.sigmoid(outputs)
                apm.add(probs.data.cpu().numpy(), (labels > 0.5).float().data.cpu().numpy())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                tot_loss += loss.data[0]

            epoch_loss = tot_loss / num_iter
            if phase == 'val' and apm.value().mean() > best_acc:
                best_acc = apm.value().mean()
                val_res = tr

            print '{} Loss: {:.4f} mAP: {:.4f}'.format(phase, epoch_loss, apm.value().mean())

inp_feat =  1024
#model = models.per_frame(inp_feat, 6)
model = models.sub_event(inp_feat, 6)
#model = models.tconv(inp_feat, 8)
#model = models.pyramid(inp_feat, 8)
#model = models.max_pool(inp_feat, 8)

# move to GPU
model.cuda()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, num_epochs=10)
