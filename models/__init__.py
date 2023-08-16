import torch
from .VAE import VAE
from .Unet import Unet, conv_fusion
from .ViT import ViT
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
    if name == "vae":
        return VAE()
    if name == "unet":
        return Unet(ch=opts.channels, step=shift_step, fusion=conv_fusion)
    if name == "mst-s":
        return MST(dim=28, stage=2, num_blocks=[2, 2, 2])