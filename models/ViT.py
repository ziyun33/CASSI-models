import torch.nn as nn
import torch
import torch.nn.functional as F


class MSA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class FFN(nn.Module):
    def __init__(self, ch, k) -> None:
        super().__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch * k, kernel_size=1)
        self.dwconv = nn.Conv2d(in_channels=ch * k, out_channels=ch * k, kernel_size=3, stride=1, padding=1, groups = ch * k)
        self.conv2 = nn.Conv2d(in_channels=ch * k, out_channels=ch, kernel_size=1)

    def forward(self, x):
        x = self.conv1
        x = self.gelu(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = self.conv2(x)

        return x


class MSA_block(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class ViT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = self.embedding(x)

        for i in len(self.MSA_blocks):
            block = self.MSA_blocks[i]
            if i == 0:
                x1 = block(x)
            else:
                x1 = block(x1)

        out = x + x1

        return out
