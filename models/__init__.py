import torch
import torch.nn as nn
from .Unet import Unet, conv_fusion, add_fusion
from .ViT import *
from .TSA_Net import TSA_Net
from .GAP_Net import GAP_net
from .ADMM_Net import ADMM_net
from .BIRNAT import BIRNAT
from .MST import MST
from .CST import CST
from .MST_Plus_Plus import MST_Plus_Plus
from .DAUHST import DAUHST

import sys
sys.path.append('..')
from config import *

def model_generator(name, shift_step=2):
    # End-to-End
    if name == "unet":
        return Unet(ch=opts.channels, layer_num=5, updown=[], dwconv=[], pconv=[], bn=True, step=shift_step, fusion=conv_fusion, activate=nn.ReLU())
    elif name == "vit":
        dim = 128
        patch_size = (4, 4)
        return ViT(GlobalMSA, linear_FFN, ToPatchEmb(img_size=(256,256), patch_size=patch_size, ch=28, dim=dim), LinearMapping(patch_size=patch_size, ch=28, dim=dim), blocknum=3, dim=dim, heads=8, dim_head=32, dropout=0.2)
    elif name == 'tsa_net':
        model = TSA_Net()
    elif name == 'birnat':
        model = BIRNAT()
    elif name == "mst-s":
        return MST(dim=28, stage=2, num_blocks=[2, 2, 2])
    elif name == 'mst_m':
        model = MST(dim=28, stage=2, num_blocks=[2, 4, 4])
    elif name == 'mst_l':
        model = MST(dim=28, stage=2, num_blocks=[4, 7, 5])
    elif name == 'mst_plus_plus':
        model = MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3)
    elif name == "cst-s":
        return CST(num_blocks=[1, 1, 2], sparse=True)
    elif name == 'cst_m':
        model = CST(num_blocks=[2, 2, 2], sparse=True)
    elif name == 'cst_l':
        model = CST(num_blocks=[2, 4, 6], sparse=True)
    elif name == 'cst_l_plus':
        model = CST(num_blocks=[2, 4, 6], sparse=False)

    # Deep Unfloding
    elif name == 'gap_net':
        model = GAP_net(stage=3, ch=28, step=2)
    elif name == 'admm_net':
        model = ADMM_net()
    elif 'dauhst' in name:
        num_iterations = int(name.split('_')[1][0])
        model = DAUHST(num_iterations=num_iterations)

    else:
        raise Exception(f'Method {name} is not defined !!!!')

    return model