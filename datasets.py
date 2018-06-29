import os

from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from scipy.spatial.distance import cdist
import numpy as np
from itertools import product


def calcRefAngles(positions, references=np.array([[0.,0.], [2.5, 2.5], [0.0,5.0]])):
    angles = []
    for idx, ref in enumerate(references):
        ang = np.rad2deg(np.arctan2(positions[:,1] - ref[1], positions[:,0] - ref[0]))
        idx = np.where(ang < 0)[0]
        ang[idx] += 360
        angles.append(ang)
    angles = np.array(angles)
    return angles/np.max(angles)
    

def calcAdHocAngles(p1, p2):
    ang = np.rad2deg(np.arctan2(p2[:,1] - p1[:,1], p2[:,0] - p1[:,0]))
    idx = np.where(ang < 0)[0]
    ang[idx] += 360
    return ang/np.max(ang)


class AdHocDataset(Dataset):
    """Ad Hoc Angles"""

    def __init__(self, train=True, seed=10, num_samps=5000):
        """
        Args:
        """
        self.train = train

        if self.train:
            self.seed = 10
        else:
            self.seed = 12
        
        self.num_samps = num_samps
        np.random.seed(self.seed)
        pos = np.random.rand
        self.positions1 = np.random.uniform(0,4.9, num_samps*2).reshape(-1,2).astype(np.float32)
        self.positions2 = np.random.uniform(0,4.9, num_samps*2).reshape(-1,2).astype(np.float32)
        self.ref_angles1 = calcRefAngles(self.positions1).astype(np.float32)
        self.ref_angles2 = calcRefAngles(self.positions2).astype(np.float32)
        self.ref_angles = np.expand_dims(np.vstack((self.ref_angles1, self.ref_angles2)), axis=1)
        self.ref_angles = np.transpose(self.ref_angles, (2,1,0))
        self.ys = np.expand_dims(
                        calcAdHocAngles(self.positions1, self.positions2),
                        axis=1).astype(np.float32)
        np.random.seed()


    def __len__(self):
        return self.ref_angles.shape[0]

    def __getitem__(self, idx):
        return  {'pos' : (self.positions1[idx], self.positions2[idx]),
                 'ref_angles': self.ref_angles[idx], 
                 'target' : self.ys[idx]}




def get_adhoc_dataset(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = AdHocDataset(train=True) 
    test = AdHocDataset(train=False, num_samps=1000)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader
