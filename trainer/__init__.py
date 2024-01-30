from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from torch.cuda.amp import autocast, GradScaler
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

class trainer():
    def __init__(self, model, dataloader, loss_fn, optimizer, scheduler, config) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.start_epoch = 0

        if config.amp == True:
            self.amp = True
            self.scaler = GradScaler()
            print("using amp")
        else:
            self.amp = False

    def train_one_epoch(self, rank=None):
        self.model.train()
        loss_list, psnr_list, ssim_list, sam_list = [], [], [], []
        for mea, mask, gt in tqdm(self.dataloader, disable=True):
            if rank is not None:
                mea, gt = mea.to(rank), gt.to(rank)
                mask = mask_to_cuda(mask, rank)
            else:
                mea, gt = mea.to(self.config.device), gt.to(self.config.device)
                mask = mask_to_cuda(mask, self.config.device)

            self.optimizer.zero_grad()
            if self.amp is True:
                with autocast():
                    output = self.model(mea, mask)
                    loss = self.loss_fn(output, gt)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_list.append(loss.item())
                psnr_list.append(psnr_(output, gt, data_range=1.0).item())
                ssim_list.append(ssim_(output, gt.to(output.dtype), data_range=1.0).item())
                sam_list.append(sam_(output, gt.to(output.dtype)).item() *  180 / np.pi)
            else:
                output = self.model(mea, mask)
                loss = self.loss_fn(output, gt)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                loss_list.append(loss.item())
                psnr_list.append(psnr_(output, gt, data_range=1.0).item())
                ssim_list.append(ssim_(output, gt, data_range=1.0).item())
                sam_list.append(sam_(output, gt).item() * 180 / np.pi)

        self.scheduler.step()

        return {"loss" : sum(loss_list) / len(loss_list), 
                "psnr" : sum(psnr_list) / len(psnr_list), 
                "ssim" : sum(ssim_list) / len(ssim_list), 
                "sam" : sum(sam_list) / len(sam_list)}

    def valid_one_epoch(self, rank=None):
        self.model.eval()
        for mea, mask, gt in tqdm(self.dataloader):
            if rank is not None:
                mea, mask, gt = mea.to(rank), mask.to(rank), gt.to(rank)
            else:
                mea, mask, gt = mea.to(self.config.device), mask.to(self.config.device), gt.to(self.config.device)
            with torch.no_grad():
                output = self.model(mea, mask)
                loss = self.loss_fn(output, gt)
        
        return loss.item()

    def train_n_epoch(self, N_epochs, rank=None):
        save_path = "checkpoints" + '/' + self.config.model_name + '/' + self.config.save_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        for i in range(self.start_epoch, N_epochs):
            # train
            info_dict = self.train_one_epoch(rank=rank)
            if rank is not None:
                dist.barrier()
            if rank is None or rank == 0:
                self.info("train", i+1, info_dict)

            # save checkpoint
            if rank is None or rank == 0:
                save_full_path =  save_path + "/" + str(i+1) + ".ckpt"
                self.save_checkpoint(i+1, save_full_path)

            dist.barrier()

    def test(self):
        self.model.eval()
        psnr_list, ssim_list, sam_list = [], [], []
        fig_path = f"figs/simu/{self.config.model_name}/{self.config.save_path}"
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        with open(f"{fig_path}/test.log", "w") as f:
            f.write(str(self.config) + "\n")

        i = 1
        for mea, mask, gt in tqdm(self.dataloader):
            mea, mask, gt = mea.to(self.config.device), mask_to_cuda(mask, self.config.device), gt.to(self.config.device)
            with torch.no_grad():
                output = self.model(mea, mask)
                psnr_list.append(psnr_(output, gt, data_range=1.0).item())
                ssim_list.append(ssim_(output, gt, data_range=1.0).item())
                sam_list.append(sam_(output, gt).item()*180/np.pi)
            
            draw_cubes(output.squeeze(0).permute(1,2,0).cpu().numpy(), list(np.linspace(450, 650, 28)), f"{fig_path}/scene_{i}.png")

            # when using 10 benchmark testing data from KAIST to test 
            if i == 9:
                corr_list = draw_line(output.squeeze(0).permute(1,2,0).cpu().numpy(), gt.squeeze(0).permute(1,2,0).cpu().numpy(), list(np.linspace(450, 650, 28)), [[40,155,55,170],[90,155,105,170],[135,155,150,170],[205,155,220,170]], fig_path)
            i = i + 1
        
        psnr_test, ssim_test, sam_test = sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(sam_list) / len(sam_list)
        with open(f"{fig_path}/test.log", "a") as f:
            for i, (p, ss, s) in enumerate(zip(psnr_list, ssim_list, sam_list)):
                print(f"[ test ] | scene {i+1:02d}: psnr: {p:.4f}, ssim: {ss:.4f}, sam: {s:.4f}")
                f.write(f"[ test ] | scene {i+1:02d}: psnr: {p:.4f}, ssim: {ss:.4f}, sam: {s:.4f}\n")
            print(f"[ test ] | average_psnr = {psnr_test:.4f}, average_ssim = {ssim_test:.4f}, average_sam = {sam_test:.4f}")
            print(f"[ test ] | average_corr = {(sum(corr_list) / len(corr_list)):.4f}")
            f.write(f"[ test ] | average_psnr = {psnr_test:.4f}, average_ssim = {ssim_test:.4f}, average_sam = {sam_test:.4f}\n")
            f.write(f"[ test ] | average_corr = {(sum(corr_list) / len(corr_list)):.4f}")

    def info(self, title, epoch, info_dict: dict):
        text = f"[ {title} | {epoch:03d}/{self.config.n_epochs:03d} ]"
        for key in info_dict.keys():
            text = text + f" {key} = {info_dict[key]:.6f},"
        text = text + f" timestamp: {timestamp()}"
        print(text)

    def save_checkpoint(self, epoch, save_path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "scheduler_state_dict" : self.scheduler.state_dict(),
            "epoch" : epoch
        }, save_path)

    def load_checkpoint(self, rank=None):
        if rank is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(self.config.checkpoint, map_location=map_location)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"]
        else:
            checkpoint = torch.load(self.config.checkpoint)
            self.model.load_state_dict(load_multigpu_paras(checkpoint["model_state_dict"]))

    def eva_FLOPs_Params(self, rank=None):

        for mea, mask, _ in self.dataloader:
            if rank is not None:
                mea, mask = mea.to(rank), mask_to_cuda(mask, rank)
            else:
                mea, mask = mea.to(self.config.device), mask_to_cuda(mask, self.config.device)

            # print(mea.shape, mask.shape)
            flops = FlopCountAnalysis(self.model, (mea, mask))
            n_param = sum([p.nelement() for p in self.model.parameters()])

            break

        print(f"FLOPs(G): {(flops.total() / (1e9)):.2f}, Params(M): {(n_param / 1e6):.2f}")
