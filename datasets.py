import os

from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from scipy.spatial.distance import cdist
import numpy as np
from itertools import product
import copy


def calcRefAngles(positions, references=np.array([[0.,0.]])):
#def calcRefAngles(positions, references=np.array([[0.,0.], [2.5, 2.5], [0.0,5.0]])):
#def calcRefAngles(positions, references=np.array([[0.,0.], [2.5, 2.5]])):
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

class OrientationDataset(Dataset):
    """Ad Hoc Orientations"""

    def __init__(self, train=True, seed=10, num_samps=5000, num_referencenodes=1, noise_scale=0.0):
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
        self.mobile_positions = np.random.uniform(0,4.9, num_samps*2).reshape(-1,2).astype(np.float32)
        
        self.references = np.array([[0., 0.], [5., 5.], [0., 5.], [5., 0.], [2.5,2.5], 
                                    [2.5, 0.], [0., 2.5], [5., 2.5], [2.5, 5.], [1.,1.]] )
        assert num_referencenodes <= 10, "Not enough reference nodes defined"
        self.references = self.references[:num_referencenodes]
        #np.random.uniform(0,4.9, num_referencenodes*2).reshape(-1,2).astype(np.float32) 
        #print(self.references)
        # angles will be normalized between 0 and 1 (lying, actually )

        self.mobile_orientations = np.random.uniform(0, 1, num_samps).astype(np.float32)         
        self.theta = calcRefAngles(self.mobile_positions, self.references).astype(np.float32) 
        self.beta = ( self.theta + self.mobile_orientations ) % 1.
        self.data = np.expand_dims(np.vstack((self.theta,self.beta)), axis=1)
        self.data = np.transpose(self.data, (2,1,0))
        self.mobile_orientations = np.expand_dims(self.mobile_orientations, axis=1)

        np.random.seed()

        noise_mean = 0.0
        gauss_noise = np.random.normal(loc=noise_mean,\
                                            scale=noise_scale,\
                                            size=(self.data.shape))

        self.data += gauss_noise


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return  {'pos' : self.mobile_positions[idx],
                 'data': self.data[idx], 
                 'target' : self.mobile_orientations[idx]}

def get_orientation_dataset(dataset_root, 
                            batch_size, 
                            num_samps, 
                            num_referencenodes, 
                            quantization, 
                            noise_scale, 
                            ref_scale, 
                            is_cuda=True):

    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = OrientationDataset(train=True, num_referencenodes=num_referencenodes, noise_scale=noise_scale) 
    test = OrientationDataset(train=False, num_referencenodes=num_referencenodes, num_samps=1000, noise_scale=noise_scale)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader




def get_adhoc_dataset(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = AdHocDataset(train=True) 
    test = AdHocDataset(train=False, num_samps=1000)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader
