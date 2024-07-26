import torch
import sys
sys.path.append('..')
 
import numpy as np
from config import cfg
from data import make_data_loader
from layers.RaySamplePoint import RaySamplePoint
from modeling import build_model
torch.cuda.set_device(3)

import random
cfg.merge_from_file('../configs/train_mnist_softmax.yml')
cfg.freeze()


train_loader, dataset = make_data_loader(cfg, is_train=True)


model = build_model(cfg).cuda()

import time

for i in train_loader:
    rays, rgbs, bboxes = i

    rays = rays[0].cuda()
    rgbs = rgbs[0].cuda()
    bboxes = bboxes[0].cuda()

    print('rays:',rays.size())
    print('rgbs:',rgbs.size())
    print('bboxes:',bboxes.size())
    


    start = time.time()
    color, depth = model( rays, bboxes)
    end = time.time()
    print(end-start)


   
    break
    