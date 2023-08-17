import os
import random
import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
from Utils import *

class HSIDataset_simu(Dataset):
    def __init__(self, hsi_path, mask_path, data_num, tfm, opt):
        super(HSIDataset_simu).__init__()

        # HSIs
        hsi_list = os.listdir(hsi_path)
        self.hsi_list_size = len(hsi_list)
        self.hsi_set = []
        for i in range(self.hsi_list_size):
            print(f'loading scene {i}')
            filename = hsi_list[i]
            path1 = os.path.join(hsi_path) + filename
            hsi = sio.loadmat(path1)
            key = list(hsi.keys())[-1]
            hsi = hsi[key]
            hsi = hsi / 65535.0
            hsi[hsi < 0] = 0
            hsi[hsi > 1] = 1
            hsi = torch.from_numpy(hsi)
            hsi = hsi.permute(2, 0, 1)
            hsi = hsi.to(torch.float32)
            self.hsi_set.append(hsi)

            dist.barrier()

        # mask 2d
        self.mask = sio.loadmat(mask_path)['mask']

        self.mask3d = np.tile(self.mask[:,:,np.newaxis], (1,1,opt.channels))
        self.mask3d = np.transpose(self.mask3d, [2, 0, 1]) # S, H, W
        self.mask3d = torch.from_numpy(self.mask3d) # tensor
        self.mask3d = self.mask3d.to(torch.float32)

        # num
        self.data_num = data_num
        self.file_num = len(hsi_list)

        # tfm
        self.tfm = tfm

        self.opt = opt
  
    def __len__(self):
        return self.data_num
  
    def __getitem__(self, idx):
        index = random.randint(0, self.hsi_list_size - 1)
        hsi = self.hsi_set[index]
        px = random.randint(0, hsi.shape[1] - self.opt.mask_size)
        py = random.randint(0, hsi.shape[2] - self.opt.mask_size)
        hsi = hsi[:, px:px + self.opt.mask_size:1, py:py + self.opt.mask_size:1] # S, H, W
        gt = self.tfm(hsi) # S, H, W

        mask = self.init_mask(self.mask3d)
        mea = self.init_mea(gt, self.mask3d)

        # measurement, mask, ground_truth
        return mea, mask, gt
    
    def init_mask(self, mask3d):
        if self.opt.mask_type == "None":
            return None
        elif self.opt.mask_type == "mask":
            return mask3d
        elif self.opt.mask_type == "Phi":
            Phi = shift(mask3d, self.opt.shift_step)
            return Phi
        elif self.opt.mask_type == "Phi_PhiPhiT":
            Phi = shift(mask3d, self.opt.shift_step)
            Phi_s =torch.sum(Phi**2, 0)
            Phi_s[Phi_s==0] = 1
            return Phi, Phi_s

    def init_mea(self, gt, mask3d):
        temp = shift(mask3d * gt, self.opt.shift_step)
        mea = torch.sum(temp, 0)
        if self.opt.mea_type == "Y":
            return mea / self.opt.channels * 2
        elif self.opt.mea_type == "H" or self.opt.mea_type == "HM":
            nC = self.opt.channels
            meas = meas / nC * 2
            H = shift_back(meas, self.opt.shift_step, nC)
            if self.opt.mea_type == "HM":
                HM = torch.mul(H, mask3d)
                return HM
            else:
                return H

        return mea

class HSIDataset_real(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
    
class HSIDataset_test(Dataset):
    def __init__(self, hsi_path, mask_path, opt):
        super(HSIDataset_simu).__init__()

        # HSIs
        hsi_list = os.listdir(hsi_path)
        hsi_list.sort()
        print(hsi_list)
        self.hsi_list_size = len(hsi_list)
        self.hsi_set = np.zeros((((opt.mask_size, opt.mask_size, opt.channels, self.hsi_list_size))))
        for i in range(self.hsi_list_size):
            print(f'loading scene {i}')
            filename = hsi_list[i]
            path1 = os.path.join(hsi_path) + filename
            hsi = sio.loadmat(path1)
            key = list(hsi.keys())[-1]
            hsi = hsi[key]

            self.hsi_set[:,:,:,i] = hsi[:,:,:]

        self.hsi_set = torch.from_numpy(self.hsi_set)
        self.hsi_set = self.hsi_set.permute(2, 0, 1, 3)
        self.hsi_set = self.hsi_set.to(torch.float32)

        # mask 2d
        self.mask = sio.loadmat(mask_path)['mask']

        self.mask3d = np.tile(self.mask[:,:,np.newaxis], (1,1,opt.channels))
        self.mask3d = np.transpose(self.mask3d, [2, 0, 1]) # S, H, W
        self.mask3d = torch.from_numpy(self.mask3d) # tensor
        self.mask3d = self.mask3d.to(torch.float32)

        self.opt = opt
  
    def __len__(self):
        return self.hsi_list_size
  
    def __getitem__(self, idx):
        gt = self.hsi_set[:, :, :, idx]

        mask = self.init_mask(self.mask3d)
        mea = self.init_mea(gt, self.mask3d)

        # measurement, mask, ground_truth
        return mea, mask, gt
    
    def init_mask(self, mask3d):
        if self.opt.mask_type == "None":
            return None
        elif self.opt.mask_type == "mask":
            return mask3d
        elif self.opt.mask_type == "Phi":
            Phi = shift(mask3d, self.opt.shift_step)
            return Phi
        elif self.opt.mask_type == "Phi_PhiPhiT":
            Phi = shift(mask3d, self.opt.shift_step)
            Phi_s =torch.sum(Phi**2, 0)
            Phi_s[Phi_s==0] = 1
            return Phi, Phi_s

    def init_mea(self, gt, mask3d):
        temp = shift(mask3d * gt, self.opt.shift_step)
        mea = torch.sum(temp, 0)
        if self.opt.mea_type == "Y":
            return mea / self.opt.channels * 2
        elif self.opt.mea_type == "H" or self.opt.mea_type == "HM":
            nC = self.opt.channels
            meas = meas / nC * 2
            H = shift_back(meas, self.opt.shift_step, nC)
            if self.opt.mea_type == "HM":
                HM = torch.mul(H, mask3d)
                return HM
            else:
                return H

        return mea


def get_dataloader(hsidataset, data_path, mask_path, data_num, tfm, opts):
    dataset = hsidataset(data_path, mask_path, data_num, tfm, opts)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=torch.cuda.device_count(), sampler=sampler)

    return loader

def get_dataloader_test(hsidataset, data_path, mask_path, opts):
    dataset = hsidataset(data_path, mask_path, opts)

    return DataLoader(dataset, batch_size=1)