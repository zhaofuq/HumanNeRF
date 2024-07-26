import sys
sys.path.append('..')
from layers.RaySamplePoint import RaySamplePoint
import torch
import numpy as np
from utils.ray_sampling import ray_sampling
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

rsp = RaySamplePoint()
# rays = torch.from_numpy(np.array([[-0.5, 0.5, 0.5, 1, 0, 0],[-0.5, 0.5, 0.5, 0.5, 0, 0.25],[0.5, -0.5, 0.5, 0, 0.5, 0.25]]).astype(np.float)).reshape((3, 6))
# bbox = np.array([[[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
#                  [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
#                  [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]]).astype(np.float)
# bbox = torch.from_numpy(bbox).reshape((3, 8, 3))
# print(bbox)
# method=None
# print(rsp.forward(rays, bbox, method))
Ks = torch.tensor( [[750.,0,1000],[0,750,1000],[0,0,1]] ).unsqueeze(0)
Ts = torch.tensor( [[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] ).unsqueeze(0)
#bbox = np.array([[[0,0,0], [1, 0, 0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]]).astype(np.float)*500 + np.array([-200,-200, 50]).reshape((-1, 1, 3))

bbox = np.array([[[-100,-100,0], [-5, -100, 0], [-5,-5,0], [-100,-5,0], [-100,-100,100], [-5, -100, 100], [-5,-5,100], [-100,-5,100]]])

image_size = (2000, 2000)
rays,rgbs = ray_sampling(Ks,Ts,image_size)
# print(bbox)
bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
sample_t,sample_point, mask= rsp.forward(rays[:5000].reshape(-1,6), bbox, method=None)
#print(sample_point[4000:4005,:])
