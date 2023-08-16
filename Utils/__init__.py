import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from .ssim_torch import ssim
from .loss_functions import SSIMLoss, MSE_SSIM

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

def shift(inputs, step):
    [nC, row, col] = inputs.shape
    output = torch.zeros(nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[i, :, step * i:step * i + col] = inputs[i, :, :]
    return output

def shift_back(inputs, step, nC):  # input [256,310]  output [28, 256, 256]
    [row, col] = inputs.shape
    output = torch.zeros(nC, row, col - (nC - 1) * step)
    for i in range(nC):
        output[i, :, :] = inputs[:, step * i:step * i + col - (nC - 1) * step]
    return output

def torch_psnr(img, ref):  # input [28,256,256]
    # img = (img*255).round()
    # ref = (ref*255).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((1)/mse).item()
    return psnr / nC

def torch_psnr_batch(img, ref):
    batch_num = img.shape[0]
    psnr_list = []

    for i in range(batch_num):
        psnr_list.append(torch_psnr(img[i,:,:,:], ref[i,:,:,:]))

    return sum(psnr_list) / len(psnr_list)

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0)).item()

def torch_ssim_batch(img, ref):
    batch_num = img.shape[0]
    ssim_list = []

    for i in range(batch_num):
        ssim_list.append(torch_ssim(img[i,:,:,:], ref[i,:,:,:]))

    return sum(ssim_list) / len(ssim_list)

def load_multigpu_paras(model_state_dict):
    # state_dict = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    
    return new_state_dict

def timestamp():
    t = time.localtime()
    result = str(t.tm_year) + str(t.tm_mon).zfill(2) + str(t.tm_mday).zfill(2) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + str(t.tm_sec).zfill(2)

    return result

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False