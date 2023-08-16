import torch
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0) -> None:
        super().__init__()
        self.data_range = data_range

    def forward(self, pred, gt) -> torch.Tensor:
        return 1 - ssim_(pred, gt, data_range=self.data_range)
    

class MSE_SSIM(nn.Module):
    def __init__(self, alpha=1, beta=0.1, data_range=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.data_range = data_range

    def forward(self, pred, gt)-> torch.Tensor:
        return self.alpha * nn.functional.mse_loss(pred, gt) + self.beta * (1 - ssim_(pred, gt, data_range=self.data_range))