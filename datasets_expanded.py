import os

from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from scipy.spatial.distance import cdist
import numpy as np
from itertools import product
import copy


def calcRefAngles(positions, references=np.array([[0.,0.], [2.5, 2.5], [0.0,5.0]])):
#def calcRefAngles(positions, references=np.array([[0.,0.], [2.5, 2.5]])):
    angles = []
    for idx, ref in enumerate(references):
        ang = np.rad2deg(np.arctan2(positions[:,1] - ref[1], positions[:,0] - ref[0]))
        idx = np.where(ang < 0)[0]
        ang[idx] += 360
        angles.append(ang)
    angles = np.array(angles)
    return angles/np.max(angles)



# calculate reference angles for random reference nodes and perform quantization on reference angles
def calcRefAnglesRandRefQuantization(positions, references, quantization):
    angles = []
    for idx, ref in enumerate(references):
        ang = np.rad2deg(np.arctan2(positions[:,1] - ref[:,1], positions[:,0] - ref[:,0]))
        idx = np.where(ang < 0)[0]
        ang[idx] += 360
        angles.append(ang)
    angles = np.array(angles)
    # angles = angles[:2,:]
    quantized_angles = copy.deepcopy(angles)
    if quantization > 0:
        quantized_angles /= float(quantization)
        quantized_angles = np.around(quantized_angles, 0)
        quantized_angles *= float(quantization)
    # print( "Num Unique: {}".format( np.unique(quantized_angles,
    # axis=1).shape))
    
    return quantized_angles/np.max(quantized_angles)


def RejectSamplingRemove(quant_angles_1, quant_angles_2, positions_u, positions_v, references, final_num_samples):
    num_refnodes = references.shape[0]
    # references = references[:2, :, :]
    print( "quant_angles_1: {}".format( quant_angles_1.shape))
    print("quant_angles_2: {}".format( quant_angles_2.shape))
    print( "positions_u: {}".format( positions_u.shape))
    print( "positions_v: {}".format( positions_v.shape))
    print( "references: {}".format( references.shape))

    # allocate arrays to hold reject sampling data
    new_quant_angles_1 = np.zeros((num_refnodes,final_num_samples))
    new_quant_angles_2 = np.zeros((num_refnodes,final_num_samples))
    new_positions_u = np.zeros((final_num_samples,2)) 
    new_positions_v = np.zeros((final_num_samples,2)) 
    new_references = np.zeros((num_refnodes,final_num_samples,2))

    # find many-to-one repeats
    unq, match_idxs = np.unique(quant_angles_1, return_inverse=True, axis=1)

    # find how many many-to-one repeats exist
    num_unique = np.max(match_idxs)

    # fill up how many samples you want to end with
    for i in range(final_num_samples):
        # get all examples of particular many-to-one
        same_quantangles_idxs = np.where(match_idxs == i)[0]

        # calculate sum of l2 distance between u node and each reference node for all examples of this many-to-one 
        l2_dist = np.zeros(len(same_quantangles_idxs))
        for j in range(len(references)):
            samepos_refs = references[j,same_quantangles_idxs,:]
            samepos_upos = positions_u[same_quantangles_idxs,:]
            l2_dist[:] += np.linalg.norm(samepos_refs - samepos_upos)

        # find the index of the many-to-one whose sum of l2 distances is closest to u
        min_l2_idx = same_quantangles_idxs[np.argmin(l2_dist)]
        
        # use that many-to-one as the sample in the training set
        new_quant_angles_1[:,i] = quant_angles_1[:,min_l2_idx]
        new_quant_angles_2[:,i] = quant_angles_2[:,min_l2_idx]
        new_positions_u[i,:] = positions_u[min_l2_idx, :]
        new_positions_v[i,:] = positions_v[min_l2_idx, :]
        new_references[:,i,:] = references[:,min_l2_idx,:]

    return new_quant_angles_1, new_quant_angles_2, new_positions_u, new_positions_v, new_references





    

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



class RandomRefDataset(Dataset):
    """Ad Hoc Angles"""

    def __init__(self, train=True, seed=10, num_samps=5000, num_referencenodes=2, 
                       quantization=0, rejection_sampling=False, noise_scale=0.0, ref_scale=1.0):
        """
        Args:
        """
        self.train = train

        if self.train:
            self.seed = 10
            if rejection_sampling:
                original_num_samps = copy.deepcopy(num_samps)
                num_samps *= 2
        else:
            self.seed = 12
        
        self.num_samps = num_samps
        
        # set random seed to be not random for generating positions
        np.random.seed(self.seed)

        # generate positions
        self.positions1 = np.random.uniform(0,4.9, num_samps*2).reshape(-1,2).astype(np.float32)
        self.positions2 = np.random.uniform(0,4.9, num_samps*2).reshape(-1,2).astype(np.float32)


        # TODO: check this
        # generate reference node positions
        self.ref_scale = ref_scale
        newmethodref_list = []
        tmp_l = []
        for i in range(num_referencenodes):
            newmethodref_list.append([np.random.normal(loc=self.positions1, scale=self.ref_scale)])
            xs = [np.random.uniform(low=x-self.ref_scale,high=x+self.ref_scale, size=1) for x in self.positions1[:,0]]
            ys = [np.random.uniform(low=y-self.ref_scale,high=y+self.ref_scale, size=1) for y in self.positions1[:,1]]
            tmp_l.append( [np.hstack((xs,ys))])
        self.newmethodref = np.array(newmethodref_list)
        # self.newmethodref = np.array(tmp_l)

        # re-set random seed to be random for generating reference nodes
        np.random.seed()


        self.newmethodref = self.newmethodref.reshape(num_referencenodes,-1,2)

        # calculate reference angles and quantize
        self.ref_angles1 = calcRefAnglesRandRefQuantization(self.positions1, self.newmethodref, quantization).astype(np.float32)
        self.ref_angles2 = calcRefAnglesRandRefQuantization(self.positions2, self.newmethodref, quantization).astype(np.float32)

        # if doing rejection sampling, reject everything but closest
        if rejection_sampling and self.train:
            new_quant_angles_1, new_quant_angles_2, new_positions_u, new_positions_v, new_references = RejectSamplingRemove(self.ref_angles1, self.ref_angles2, self.positions1, self.positions2, self.newmethodref, original_num_samps)
            self.ref_angles1 = new_quant_angles_1.astype(np.float32)
            self.ref_angles2 = new_quant_angles_2.astype(np.float32)
            self.positions1 = new_positions_u.astype(np.float32)
            self.positions2 = new_positions_v.astype(np.float32)
            self.newmethodref = new_references


        # prepare data for use with neural net
        self.ref_angles = np.expand_dims(np.vstack((self.ref_angles1, self.ref_angles2)), axis=1)
        self.ref_angles = np.transpose(self.ref_angles, (2,1,0))


        # add noise
        # TODO: parameterize this from command line
        self.noise_scale = noise_scale 
        if self.noise_scale > 0.0:
            self.ref_angles += np.random.normal(loc=0, scale=self.noise_scale, size=self.ref_angles.shape)
        

        self.ys = np.expand_dims(
                        calcAdHocAngles(self.positions1, self.positions2),
                        axis=1).astype(np.float32)
        


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

def get_randomref_dataset(dataset_root, batch_size, num_samps, num_referencenodes, quantization, rejection_sampling, noise_scale, ref_scale, is_cuda=True):
    kwargs = {'num_workers': 12, 'pin_memory': True} if is_cuda else {}
    train = RandomRefDataset(train=True, 
                             num_samps=num_samps['train'], 
                             num_referencenodes=num_referencenodes, 
                             quantization=quantization, 
                             rejection_sampling=rejection_sampling,
                             noise_scale=noise_scale,
                             ref_scale=ref_scale) 

    test = RandomRefDataset(train=False, 
                            num_samps=num_samps['test'], 
                            num_referencenodes=num_referencenodes, 
                            quantization=quantization, 
                            rejection_sampling=rejection_sampling,
                            noise_scale=noise_scale,
                            ref_scale=ref_scale)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=True, **kwargs)
    
    return train, train_loader, test, test_loader
