
import argparse
import os
import sys
from os import mkdir
from apex import amp
import shutil




import torch.nn.functional as F

sys.path.append('..')
from config import cfg
from data import make_data_loader, make_data_loader_view
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
import torch
from layers.RaySamplePoint import RaySamplePoint
import random
import time

torch.cuda.set_device(int(sys.argv[1]))



cfg.merge_from_file('../configs/config.yml')
cfg.freeze()



train_loader, dataset = make_data_loader(cfg, is_train=True)
val_loader, dataset_val = make_data_loader_view(cfg, is_train=True)
model = build_model(cfg).cuda()

maxs = torch.max(dataset.bbox[0], dim=0).values.cuda()+0.5
mins = torch.min(dataset.bbox[0], dim=0).values.cuda()-0.5
model.set_max_min(maxs,mins)


optimizer = make_optimizer(cfg, model)

scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)



loss_fn = make_loss(cfg)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")



beg = time.time()
for batch in train_loader:
    beg = time.time()
    model.train()
    optimizer.zero_grad()


    rays, rgbs, bboxes = batch

    rays = rays[0].cuda()
    rgbs = rgbs[0].cuda()
    bboxes = bboxes[0].cuda()


    stage2, stage1 = model( rays, bboxes, only_coarse = False)
    print('forward:',time.time()-beg)

    loss1 = loss_fn(stage2[0], rgbs)
    loss2 = loss_fn(stage1[0], rgbs)

    loss = loss1 + loss2


    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    #loss.backward()

    optimizer.step()


    loss.item()
    break

print('forward+backward:', time.time()-beg)

print( '%f rays/s' % (float(cfg.SOLVER.BUNCH) / (time.time()-beg)))