import torch
import torch.nn as nn

from utils import Trigonometric_kernel
class DeltaNet(nn.Module):

    def __init__(self,  c_dist = 24, c_rdir = 72, feat_dim = 128 ,include_input = True):
        """ Init layered sampling
        """
        super(DeltaNet, self).__init__()
        self.c_dist = c_dist
        self.c_rdir = c_rdir

        #Positional Encoding
        self.tri_kernel_dist = Trigonometric_kernel(L=4,input_dims = c_dist,include_input = include_input, use_weights = True)

        self.rel_dist_dim = self.tri_kernel_dist.calc_dim(c_dist)
        
        self.output_dim = 3
        self.feat_dim = feat_dim

        backbone_dim = 256
        head_dim = 128
        
        self.delta_net = nn.Sequential(
            nn.Linear(self.rel_dist_dim + self.c_rdir + self.feat_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim,self.output_dim),
        )
        
    def forward(self, dist, rdir, feats):
        """ Generate sample points
        Input:
        pos: [N,3] points in real world coordinates

        Output:
        weights: [N,24]  bone weights for the pos
        """

        bins_mode = False
        if len(dist.size()) > 2:
            bins_mode = True
            L = dist.size(1)
            dist = dist.reshape((-1,self.c_dist))     #(N,c_pos)
            rdir = rdir.reshape((-1,self.c_rdir))
            feats = feats.reshape((-1,self.feat_dim))
            
        dist = self.tri_kernel_dist(dist)

        offset = self.delta_net(torch.cat([dist, rdir, feats],dim=-1))

        if bins_mode:
            offset = offset.reshape((-1, L, self.output_dim))
            
        return offset
