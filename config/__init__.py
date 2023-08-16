import argparse
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate

# from .models import *
# from .data import *


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
parser.add_argument('--origin-data-size', type=int,
                        default=1024, help='size of dataset')
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
parser.add_argument('--device', type=str,
                        default='cuda', help='training device')
parser.add_argument('--gpu-ids', '-gi', type=str, 
                        default='7,8', help='gpu ids')

# configs about training
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
parser.add_argument('--amp', type=str, default="True",
                        help="auto mix precision")

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
    RandomRotation90(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

test_tfm = transforms.Compose([
    transforms.ToTensor()
])
