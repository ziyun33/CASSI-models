import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_


class RMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, gt):
        return torch.sqrt(F.mse_loss(pred, gt))

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
        return self.alpha * F.mse_loss(pred, gt) + self.beta * (1 - ssim_(pred, gt, data_range=self.data_range))
    
class MSE_Sparsity(nn.Module):
    def __init__(self, Lambda=2) -> None:
        super().__init__()
        self.Lambda = Lambda

    def forward(self, pred, mask, gt):
        mseloss = F.mse_loss(pred, gt)
        sparsityloss = F.mse_loss(mask, torch.mean(torch.abs(pred - gt), dim=1, keepdim=True))
        return mseloss + self.Lambda * sparsityloss