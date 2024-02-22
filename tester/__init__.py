from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import io as sio
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_
from torchmetrics.functional.image import spectral_angle_mapper as sam_

import sys
sys.path.append("..")
from Utils import *
from visualizer import *

def mask_to_cuda(mask, device):
    if mask == None:
        return None
    elif type(mask) == tuple or type(mask) == list:
        return (mask[0].to(device), mask[1].to(device))
    else:
        return mask.to(device)

class tester():
    def __init__(self, model, dataloader, config) -> None:
        self.model = model
        self.dataloader = dataloader
        self.config = config

    def test(self):
        self.model.eval()
        psnr_list, ssim_list, sam_list = [], [], []
        result_path = f"result/simu/{self.config.model_name}/{self.config.save_path}"
        Path(result_path).mkdir(parents=True, exist_ok=True)

        outputs = []
        i = 1

        for mea, mask, gt in tqdm(self.dataloader):
            mea, mask, gt = mea.to(self.config.device), mask_to_cuda(mask, self.config.device), gt.to(self.config.device)
            with torch.no_grad():
                output = self.model(mea, mask)
                outputs.append(output)
                psnr_list.append(psnr_(output, gt, data_range=1.0).item())
                ssim_list.append(ssim_(output, gt, data_range=1.0).item())
                sam_list.append(sam_(output, gt).item()*180/np.pi)
            
            draw_cubes(output.squeeze(0).permute(1,2,0).cpu().numpy(), list(np.linspace(450, 650, 28)), f"{result_path}/scene_{i}.png")

            i = i + 1

        self.logger(result_path, (psnr_list, ssim_list, sam_list))
        self.save_result(outputs=outputs, path=result_path)

    def logger(self, path, info):
        psnr_list, ssim_list, sam_list = info
        psnr_test = sum(psnr_list) / len(psnr_list)
        ssim_test = sum(ssim_list) / len(ssim_list)
        sam_test = sum(sam_list) / len(sam_list)
        with open(f"{path}/test.log", "w") as f:
            f.write(str(self.config) + "\n")

            for i, (p, ss, s) in enumerate(zip(psnr_list, ssim_list, sam_list)):
                print(f"[ test ] | scene {i+1:02d}: psnr: {p:.4f}, ssim: {ss:.4f}, sam: {s:.4f}")
                f.write(f"[ test ] | scene {i+1:02d}: psnr: {p:.4f}, ssim: {ss:.4f}, sam: {s:.4f}\n")
            print(f"[ test ] | average_psnr = {psnr_test:.4f}, average_ssim = {ssim_test:.4f}, average_sam = {sam_test:.4f}")
            f.write(f"[ test ] | average_psnr = {psnr_test:.4f}, average_ssim = {ssim_test:.4f}, average_sam = {sam_test:.4f}\n")

    def save_result(self, outputs, path):
        for i, output in enumerate(outputs):
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sio.savemat(path + f"/scene{i+1:02d}.mat", mdict={'img':output})

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint)
        self.model.load_state_dict(load_multigpu_paras(checkpoint["model_state_dict"]))

    def eva_FLOPs_Params(self):

        for mea, mask, _ in self.dataloader:
            mea, mask = mea.to(self.config.device), mask_to_cuda(mask, self.config.device)

            # print(mea.shape, mask.shape)
            flops = FlopCountAnalysis(self.model, (mea, mask))
            n_param = sum([p.nelement() for p in self.model.parameters()])

            break

        print(f"FLOPs(G): {(flops.total() / (1e9)):.2f}, Params(M): {(n_param / 1e6):.2f}")
