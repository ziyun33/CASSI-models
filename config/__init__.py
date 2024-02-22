import argparse
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
from .loss_functions import *

def str2bool(s: str):
    return True if s.lower() == "true" else False

parser = argparse.ArgumentParser(
        description='config')

# configs about mode
parser.add_argument('--mode', '-m', type=str, 
                        default='train', help='train or test')

# configs about data
parser.add_argument('--train-data-root', type=str,
                        default='data/cave_1024_28/', help='train data root')
parser.add_argument('--valid-data-root', type=str,
                        default='valid_data/', help='valid data root')
parser.add_argument('--test-data-root', type=str,
                        default='data/test_benchmark/Truth/', help='test data root')
parser.add_argument('--train-data-num', type=int,
                        default=2000, help='size of dataset')
parser.add_argument('--valid-data-num', type=int,
                        default=1000, help='size of dataset')
parser.add_argument('--channels', type=int,
                        default=28, help='channels of HSI')
parser.add_argument('--mea-type', type=str,
                        default="Y", help='type of measurement')

# config about mask
parser.add_argument('--mask-path', type=str,
                        default='data/masks/mask.mat', help='mask path')
parser.add_argument('--shift-step', type=int,
                        default=2, help='shift step')
parser.add_argument('--mask-type', type=str,
                        default='mask', help='type of mask')
parser.add_argument('--mask-size', type=int,
                        default=256, help="size of mask")

# configs about model
parser.add_argument('--model-name', type=str,
                        default='unet', help='model name')
parser.add_argument('--checkpoint', '-cp', type=str,
                        default=None, help='checkpoint to use.')
parser.add_argument('--save-path', '-sp', type=str,
                        default='checkpoints', help='path to save checkpoints')

# configs about device
parser.add_argument('--gpu-ids', '-gi', type=str, 
                        default='0,1', help='gpu ids')
parser.add_argument('--port', type=str,
                        default="12345", help="ddp port")
parser.add_argument('--device', type=str,
                        default='cuda', help='training device')

# configs about training
parser.add_argument('--loss-fn', type=str,
                        default="mse", help="loss function")
parser.add_argument('--scheduler', type=str, 
                        default="CosineAnnealingLR", help="scheduler")
parser.add_argument('--n-epochs', type=int,
                        default=100, help='epochs of training')
parser.add_argument('--batch-size', '-b', type=int,
                        default=16, help='training batch size. default=32')         
parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate. default=1e-4.')
parser.add_argument('--step', type=int, default=10,
                        help='weight decay step. default=10')
parser.add_argument('--wd', type=float, default=0.1,
                        help='weight decay. default=0.1')
parser.add_argument('--patience', type=int, default=120,
                        help='patience. default=20')
parser.add_argument('--seed', type=int, default=2023,
                        help='random seed to use. default=2023')

parser.add_argument('--auto-lr', type=str2bool, default="false",
                        help='use auto learning rate')
parser.add_argument('--amp', type=str2bool, default="true",
                        help="auto mix precision")
parser.add_argument('--compile', type=str2bool, default="true",
                        help="compile model (PyTorch >= 2.0)")

parser.add_argument('--spectral-test', type=str2bool, default="true",
                        help="draw spectral curve")

# about tqdm
bar_disable = True

opts = parser.parse_args()

class RandomRotation90:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return rotate(img, angle)

train_tfm = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.RandomCrop(opts.mask_size),
    # transforms.RandomResizedCrop(opts.mask_size, antialias=True),
    RandomRotation90(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

test_tfm = transforms.Compose([
    transforms.ToTensor()
])

def get_loss(name="mse"):
    if name == "mse":
        return nn.MSELoss()
    elif name == "rmse":
        return RMSELoss()
    elif name == "mse_ssim":
        return MSE_SSIM()
    elif name == "mse_sparsity":
        return MSE_Sparsity()
    else:
        raise Exception(f"{name} not implemented!")
    
def get_scheduler(optimizer, name="MultiStepLR"):
    if name == 'MultiStepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step, gamma=1-opts.wd)
    elif opts.scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.n_epochs, eta_min=1e-6)
    else:
        raise Exception("No scheduler")