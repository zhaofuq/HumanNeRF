import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Attention, Mlp

class AttenFeatNet(nn.Module):

    def __init__(self, c_angle = 6, c_rdir = 3, feat_dim = 128, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """ Init layered sampling
        """
        super(AttenFeatNet, self).__init__()
        self.c_angle = c_angle
        self.c_rdir = c_rdir
        self.feat_dim = feat_dim 

        self.in_dim = feat_dim + c_rdir + 1
        self.out_dim = feat_dim + c_rdir + 1

        self.norm1 = nn.LayerNorm(self.in_dim)
        self.mlp = Mlp(in_features=self.in_dim, hidden_features = 256, out_features = self.out_dim,  act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.in_dim)
        self.attention = Attention(dim=self.in_dim,num_heads=6)
        
    def forward(self, feats, dirs, angle):

        dirs = dirs[...,0:3]
        dirs = dirs/torch.norm(dirs, dim=-1, keepdim = True)

        angle = angle/torch.norm(angle, dim=-1, keepdim = True)
        angle = torch.sum(dirs[:,None,:] * angle.reshape(-1, self.c_angle, 3), dim=-1, keepdim=True)

        dirs = dirs.unsqueeze(1).repeat(1, self.c_angle, 1)
        inputs = torch.cat([feats, dirs, angle], dim=-1)

        x = self.mlp(self.norm1(inputs))
        out_feats = self.attention(self.norm2(x))
            
        return out_feats


if __name__ == "__main__":
    sample_feats = torch.ones((100,6,32))
    sample_angle = torch.ones((100,6,3))
    sample_dirs = torch.ones((100,3))

    att_model = BlendNet(c_angle = 6, c_rdir = 3, feat_dim = 32)
    sample_output = att_model(sample_feats, sample_dirs, sample_angle)
    sample_output = torch.sum(sample_output, dim=1) / 6

    print(sample_output[0])
