import numpy as np
import os 
import torch
from pytorch3d.ops import knn_points

import time

class SMPLBody:
    def __init__(self, root, threshold=0.05, k=1):
        self.root_dir = root
        self.threshold = threshold
        self.k = k
        
        self.weights = np.load(os.path.join(self.root_dir, 'weights.npy'))
        self.weights = torch.from_numpy(self.weights.astype(np.float32)).cuda()

    def get_vertex_num(self):
        return self.weights.shape[0]
        
    def set_threshold(self,threshold):
        self.threshold = threshold

    def set_k(self, k):
        self.k = k 

    def query_weights(self, pts, vertices):

        N, L = pts.shape[:2] 
        ret = knn_points(pts.reshape(N*L, 3).unsqueeze(0), vertices.unsqueeze(0), None, None, self.k)
        dist, idx = ret[0].squeeze(-1) ,ret[1].squeeze(-1)

        dist = dist.reshape(N, L)
        idx = idx.reshape(N, L)

        weights = self.weights[idx]
        mask = dist < self.threshold
        weights[~mask] = 0.

        return weights , mask

