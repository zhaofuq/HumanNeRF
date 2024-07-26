import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Trigonometric_kernel


class BlendNet(nn.Module):

    def __init__(self, c_angle = 6, c_rdir = 3, feat_dim = 128*6 ,include_input = True):
        """ Init layered sampling
        """
        super(BlendNet, self).__init__()
        self.c_angle =c_angle
        self.c_rdir = c_rdir

        #Positional Encoding
        self.tri_kernel_angle = Trigonometric_kernel(L=4,input_dims = c_angle,include_input = include_input, use_weights = True)
        self.tri_kernel_dir = Trigonometric_kernel(L=4,input_dims = c_rdir, include_input = include_input, use_weights = True)
        self.angle_dim = self.tri_kernel_angle.calc_dim(c_angle)
        self.cdir_dim = self.tri_kernel_dir.calc_dim(c_rdir)
        
        self.feat_dim = feat_dim
        self.output_dim = 6

        backbone_dim = 256
        head_dim = 128

        self.blend_net = nn.Sequential(
            nn.Linear(self.cdir_dim + self.angle_dim + self.feat_dim, backbone_dim),
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

        
    def forward(self, angle, rdir, feats):
        """ Generate sample points
        Input:
        pos: [N,3] points in real world coordinates

        Output:
        weights: [N,24]  bone weights for the pos
        """

        dirs = rdir[...,0:3]
        dirs = dirs/torch.norm(dirs, dim=-1, keepdim = True)

        bins_mode = False
        if len(angle.size()) > 2:
            bins_mode = True
            L = angle.size(1)
            angle = angle.reshape((-1,self.c_angle))     #(N,c_pos)
            rdir = rdir.reshape((-1,self.c_rdir))
            feats = feats.reshape((-1,self.feat_dim))
            
        
        angle_delta = torch.sum(dirs[:,None,:] * angle.reshape(-1, self.c_angle, 3), dim=-1)
        angle_delta = self.tri_kernel_angle(angle_delta)

        dirs = self.tri_kernel_dir(dirs)
        weights = self.blend_net(torch.cat([dirs, angle_delta,  feats],dim=-1))
        weights = F.softmax(weights, dim=1)

        if bins_mode:
            weights = weights.reshape((-1, L, self.output_dim))
            
        return weights


if __name__ == "__main__":
    sample_input = torch.ones((100,6,128))
    rayTransformer = RayTransformer(sample_num=6, embed_dim=128, out_dim=128)
    sample_output = rayTransformer(sample_input)
    sample_output = torch.sum(sample_output, dim=1) / 6
    print(sample_output.shape)
