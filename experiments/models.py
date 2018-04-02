from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np

def compute_pad(stride, k, s):
    if s % stride == 0:
        return max(k - stride, 0)
    else:
        return max(k - (s % stride), 0)


class TSF(nn.Module):

    def __init__(self, N=3, mx=False):
        super(TSF, self).__init__()

        self.N = float(N)
        self.Ni = int(N)
        self.mx = mx

        # create parameteres for center and delta of this super event
        self.center = nn.Parameter(torch.FloatTensor(N))
        self.delta = nn.Parameter(torch.FloatTensor(N))
        self.gamma = nn.Parameter(torch.FloatTensor(N))

        # init them around 0
        #self.center.data = torch.FloatTensor([-0.7, 0., 0.8])
        #self.gamma.data = torch.FloatTensor([0.00001, 0.2, 0.05])
        self.center.data.normal_(0,0.5)
        self.delta.data.normal_(0,0.01)
        self.gamma.data.normal_(0, 0.0001)


    def get_filters(self, delta, gamma, center, length, time):
        """
            delta (batch,) in [-1, 1]
            center (batch,) in [-1, 1]
            gamma (batch,) in [-1, 1]
            length (batch,) of ints
        """

        # scale to length of videos
        centers = (length - 1) * (center + 1) / 2.0
        deltas = length * (1.0 - torch.abs(delta))

        gammas = torch.exp(1.5 - 2.0 * torch.abs(gamma))
        
        a = Variable(torch.zeros(self.Ni))
        a = a.cuda()
        
        # stride and center
        a = deltas[:, None] * a[None, :]
        a = centers[:, None] + a

        b = Variable(torch.arange(0, time))
        b = b.cuda()
        
        f = b - a[:, :, None]
        f = f / gammas[:, None, None]
        
        f = f ** 2.0
        f += 1.0
        f = np.pi * gammas[:, None, None] * f
        f = 1.0/f
        f = f/(torch.sum(f, dim=2) + 1e-6)[:,:,None]

        f = f[:,0,:].contiguous()

        f = f.view(-1, self.Ni, time)
        #f = f.data.cpu().numpy()
        
        return f

    def forward(self, inp):
        video, length = inp
        batch, channels, time = video.squeeze(3).squeeze(3).size()
        # vid is (B x C x T)
        vid = video.view(batch*channels, time, 1).unsqueeze(2)
        # f is (B x T x N)
        f = self.get_filters(torch.tanh(self.delta).repeat(batch), torch.tanh(self.gamma).repeat(batch), torch.tanh(self.center.repeat(batch)), length.view(batch,1).repeat(1,self.Ni).view(-1), time)
        # repeat over channels
        f = f.unsqueeze(1).repeat(1, channels, 1, 1)
        f = f.view(batch*channels, self.Ni, time)

        # o is (B x C x N)
        o = torch.bmm(f, vid.squeeze(2))
        del f
        del vid
        o = o.view(batch, channels, self.Ni).unsqueeze(3).unsqueeze(3)
        # return (B x C(*N=1 max-pooled) x 1 x 1 x 1)
        if self.mx:
            return torch.max(o.view(-1, channels, self.Ni, 1), dim=2)[0]
        return o.view(-1, channels*self.Ni, 1)




class SubConv(TSF):
    """
    Subevents as temporal conv
    """
    def __init__(self, inp, num_f,  length):
        super(SubConv, self).__init__(num_f)
        
        self.inp = inp
        self.length = length

    
    def forward(self, x):
        # overwrite the forward pass to get the TSF as conv kernels
        t = x.size(2)
        k = super(SubConv, self).get_filters(torch.tanh(self.delta), torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)
        k = k.squeeze().unsqueeze(1).unsqueeze(1)#.repeat(1, 1, self.inp, 1)
        p = compute_pad(1, self.length, t)
        pad_f = p // 2
        pad_b = p - pad_f
        x = F.pad(x, (pad_f, pad_b)).unsqueeze(1)
        return F.conv2d(x, k).squeeze(1)

class ContSubConv(nn.Module):

    def __init__(self, inp, num_f, length, classes):
        super(ContSubConv, self).__init__()
        
        self.sub_event = SubConv(inp, num_f, length)
        self.classify = nn.Conv1d(num_f*inp, classes, 1)
        self.dropout = nn.Dropout()
        
        self.inp = inp
        self.num_f = num_f
        self.classes = classes

    def forward(self, inp):
        val = False
        dim = 1
        f = inp[0].squeeze()
        if inp[0].size()[0] == 1:
            val = True
            dim = 0
            f = f.unsqueeze(0)
        
        sub_event = self.dropout(self.sub_event(f)).view(-1, self.num_f*self.inp, f.size(2))
        cls = F.relu(sub_event)
        return self.classify(cls)


    
class TConv(nn.Module):

    def __init__(self, inp, classes):
        super(TConv, self).__init__()
        self.tconv = nn.Conv1d(inp, 512, 5, padding=2)
        self.cls = nn.Conv1d(512, classes, 1)
    
    def forward(self, x, lens):
        if x.size(0) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        t = x.size(2)
        if t < 10:
            pad = (10-t+1)//2
            x = F.pad(x, (pad, pad))
        x = self.tconv(x)
        #lens = lens.view(-1,1,1).expand(-1,512,1)
        #x = (torch.max(x, dim=2)[0].unsqueeze(2))#/lens)
        x = self.cls(x)
        return x


class Pyramid(nn.Module):

    def __init__(self, inp, classes):
        super(Pyramid, self).__init__()
        self.mp1 = nn.MaxPool1d(3,1,1)
        self.mp2 = nn.MaxPool1d(5,1,2)
        self.mp3 = nn.MaxPool1d(7,1,3)

        self.tconv = nn.Conv1d(3*inp, 512, 5, padding=2)
        self.cls = nn.Conv1d(512, classes, 1)
    
    def forward(self, x, lens):
        if x.size(0) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        t = x.size(2)

        r1 = self.mp1(x)
        r2 = self.mp2(x)
        r3 = self.mp3(x)

        x = torch.cat([r1[:,:1],r1[:,1:],r2[:,:1],r2[:,1:],r3[:,:1],r3[:,1:]], dim=1)

        x = self.tconv(x)
        #lens = lens.view(-1,1,1).expand(-1,512,1)
        #x = (torch.max(x, dim=2)[0].unsqueeze(2))#/lens)
        x = self.cls(x)
        return x


    
def baseline(inp=1024, classes=1):
    model = nn.Sequential(nn.Dropout(0.5),
                          nn.Conv3d(inp, classes, (1,1,1)))

    return model


def sub_event(inp=1024, classes=1):
    model = nn.Sequential(TSF(N=8),
                          nn.Dropout(0.5),
                          nn.Conv1d(inp*8, 512, 1),
                          nn.ReLU(),
                          nn.Conv1d(512, classes, 1))
    return model


def cont_sub_event(inp=1024, classes=8):
    model = ContSubConv(inp, 8, 5, classes)
    return model

def tconv(inp=1024, classes=1):
    model = TConv(inp, classes)

    return model

def max_pool(inp, classes):
    model = nn.Sequential(nn.MaxPool1d(5,1,2),
                          nn.Conv1d(inp, classes, 1))
    return model
def pyramid(inp, classes):
    model = Pyramid(inp, classes)
    return model
