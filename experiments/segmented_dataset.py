import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from skimage import io

import numpy as np
import json
import random

import os
import os.path


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


l2i = {'ball':0,'swing':1,'strike':2,'hit':4,'foul':4,'in play':5,'bunt':6,'hit by pitch':7}
class SegmentedPitchResultMultiLabel(data_utl.Dataset):
    """
      0 - ball
      1 - swing
      2 - strike
      3 - hit
      4 - foul
      5 - in play
      6 - bunt
      7 - hit by pitch
    """

    def __init__(self, positive, negative, split, root):
        self.root = root
        with open(positive, 'r') as f: 
            self.act_dict = json.load(f)
            for a in self.act_dict.keys():
                if self.act_dict[a]['subset'] != split:
                    del self.act_dict[a]
            
        with open(negative, 'r') as f:
            self.negs = json.load(f)
        for n in self.negs.keys():
            self.negs[n]['labels'] = []
            self.act_dict[n] = self.negs[n]

        self.videos = self.act_dict.keys()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        name = self.videos[index]
        label = self.act_dict[name]['labels']

        path = os.path.join(self.root, name+'.npy')
        multilabel = np.zeros((8,))

        img = np.load(path)
        if img.shape[-1] == 1024:
            feat = img[np.random.randint(0,10)]
        if len(label) > 0:
            for labs in label:
                multilabel[l2i[labs]] = 1
        
        return feat, multilabel, name

    def __len__(self):
        return len(self.videos)

