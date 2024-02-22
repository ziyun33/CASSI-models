import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from prodigyopt import Prodigy
import os
import time

from models import model_generator
from data import HSIDataset_simu, HSIDataset_real, HSIDataset_test, get_dataloader, get_dataloader_test
from config import *
from Utils import *
from trainer import *
from tester import tester

def train_ddp(rank, world_size):
    t1 = time.time()
    print(f"Running basic DDP example on rank {rank}.")
    ddp_setup(rank, world_size, port=opts.port)
    set_seed(opts.seed + rank)

    # model
    model = model_generator(opts.model_name, opts.shift_step)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    if(int(torch.__version__[0]) >= 2) and opts.compile is True:
        ddp_model = torch.compile(ddp_model)

    # loss
    loss = get_loss(opts.loss_fn).to(rank)

    # optimizer
    if opts.auto_lr is True:
        optimizer = Prodigy(ddp_model.parameters(), lr=1.0, weight_decay=0.01, safeguard_warmup=True, use_bias_correction=True)
        print("using auto-lr")
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), opts.lr)

    # scheduler
    scheduler = get_scheduler(optimizer, name=opts.scheduler)
    
    # dataloader
    train_loader = get_dataloader(HSIDataset_simu, opts.train_data_root, opts.mask_path, opts.train_data_num, train_tfm, opts)

    # initialize trainer
    t2 = time.time()
    print(f"Initialization cost {t2-t1} s.")
    print(f"timestamp : {timestamp()}")
    trainer_ddp = trainer(model=ddp_model, dataloader=train_loader, loss_fn=loss, optimizer=optimizer, scheduler=scheduler, config=opts)
    
    # load checkpoint
    if opts.checkpoint is not None:
        trainer_ddp.load_checkpoint(rank)

    # training
    trainer_ddp.train_n_epoch(opts.n_epochs, rank)

    ddp_cleanup()
    print("done")

def test():
    model = model_generator(opts.model_name)
    model = model.to(opts.device)
    test_loader = get_dataloader_test(HSIDataset_test, opts.test_data_root, opts.mask_path, opts)
    tester_= tester(model=model, dataloader=test_loader, config=opts)
    tester_.load_checkpoint()
    tester_.eva_FLOPs_Params()
    tester_.test()

def main():
    print(opts)
    if opts.mode == "train":
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
        n_gpus = torch.cuda.device_count()
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
