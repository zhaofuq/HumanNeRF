import torch
import sys
sys.path.append('..')
 
import numpy as np
from config import cfg
from data import make_data_loader
from layers.RaySamplePoint import RaySamplePoint
torch.cuda.set_device(3)

import random
cfg.merge_from_file('../configs/train_mnist_softmax.yml')
cfg.freeze()


train_loader, dataset = make_data_loader(cfg, is_train=True)

max_xyz = torch.max(dataset.vs, dim=0)[0]
print(max_xyz)
min_xyz = torch.min(dataset.vs, dim=0)[0]
print(min_xyz)
minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
print('bbox', bbox)
rsp = RaySamplePoint()
bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
    
point3d = []
for i in train_loader:
    rays, rgbs = i
    rays1 = rays.numpy()
    rays = rays.reshape((-1,6))
    rays1 = rays1.reshape((-1,6))
    #print(rays.shape)
    for k in range(rays.shape[0]):
        #print(k)
        color = (255,  0,0)
        for j in range(100):
            p3d = rays[k,:3]*j *0.1 + rays[k,3:]
            #print([p3d[0], p3d[1], p3d[2], color[0], color[1], color[2]])
            point3d.append([p3d[0], p3d[1], p3d[2], color[0], color[1], color[2]])
    sample_t, sample_point = rsp.forward(rays.cpu().reshape(-1,6), bbox, method=None)
    #sample_point = sample_t.reshape((-1,100,1))*rays[:,:3].reshape((-1,1,3)) + rays[:,3:].reshape((-1,1,3))
    for k in range(sample_point.shape[0]):
        for j in range(100):
            point3d.append([sample_point[k,j,0], sample_point[k,j,1], sample_point[k,j,2], 0,255,255])
    np.savetxt('./pointCloud.txt', np.array(point3d))
    #print(sample_t[500])
    break
    