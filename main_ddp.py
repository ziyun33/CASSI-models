import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

from models import model_generator
from data import HSIDataset_simu, HSIDataset_real, HSIDataset_test, get_dataloader, get_dataloader_test
from config import *
from Utils import *
from trainer import *

def train_ddp(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    ddp_setup(rank, world_size)
    set_seed(opts.seed + rank)

    # model, loss, optimizer, scheduler, dataloader
    model = model_generator(opts.model_name, opts.shift_step)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss = nn.MSELoss().to(rank)
    # loss = MSE_SSIM(alpha=1.0, beta=0.01).to(rank)
    optimizer = torch.optim.Adam(ddp_model.parameters(), opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step, gamma=1-opts.wd)
    train_loader = get_dataloader(HSIDataset_simu, opts.train_data_root, opts.mask_path, opts.train_data_num, train_tfm, opts)

    # initialize trainer
    trainer_ddp = trainer(model=ddp_model, dataloader=train_loader, loss_fn=loss, optimizer=optimizer, scheduler=scheduler, config=opts)
    
    # load checkpoint
    if opts.checkpoint is not None:
        trainer_ddp.load_checkpoint(rank)

    # eva FLOPS, PARAMS
    # if rank == 0:
    #     trainer_ddp.eva_FLOPs_Params(rank)

    # training
    trainer_ddp.train_n_epoch(opts.n_epochs, rank)

    ddp_cleanup()
    print("done")

def test():
    model = model_generator(opts.model_name)
    model = model.to("cuda")
    test_loader = get_dataloader_test(HSIDataset_test, opts.test_data_root, opts.mask_path, opts)
    tester= trainer(model=model, dataloader=test_loader, loss_fn=None, optimizer=None, scheduler=None, config=opts)
    tester.load_checkpoint()
    tester.eva_FLOPs_Params()
    tester.test()

def main():
    print(opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    n_gpus = torch.cuda.device_count()
    if opts.mode == "train":
        print(n_gpus)
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        mp.spawn(train_ddp,
                args=(world_size,),
                nprocs=world_size,
                join=True)
    else:
        test()

if __name__ == "__main__":
    main()
