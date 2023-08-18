import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

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

def load_multigpu_paras(model_state_dict):
    # state_dict = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if(int(torch.__version__[0]) >= 2):
            name = k[17:]
        else:
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
    torch.backends.cudnn.benchmark = True