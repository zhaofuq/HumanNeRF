import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np

# World Coordinate Transformation Layer
class TransformLayer(nn.Module):

    def __init__(self):
        """ Init layered sampling
        """
        super(TransformLayer, self).__init__()
        
        identity_quat = torch.Tensor([0, 0, 0, 1])
        identity_off = torch.Tensor([0, 0, 0])

        self.rvec = nn.Parameter(torch.Tensor(identity_quat)) # [N_cameras, 4]
        self.tvec = nn.Parameter(torch.Tensor(identity_off))  # [N_cameras, 3]

    def rot_mats(self):
        #Quaternion->Rotation Matrix
        #theta = torch.sqrt(1e-5 + torch.sum(self.rvec ** 2))
        theta = torch.norm(self.rvec)
        rvec = self.rvec / theta
        return torch.stack((
            1. - 2. * rvec[1] ** 2 - 2. * rvec[2] ** 2,
            2. * (rvec[0] * rvec[1] - rvec[2] * rvec[3]),
            2. * (rvec[0] * rvec[2] + rvec[1] * rvec[3]),

            2. * (rvec[0] * rvec[1] + rvec[2] * rvec[3]),
            1. - 2. * rvec[0] ** 2 - 2. * rvec[2] ** 2,
            2. * (rvec[1] * rvec[2] - rvec[0] * rvec[3]),

            2. * (rvec[0] * rvec[2] - rvec[1] * rvec[3]),
            2. * (rvec[0] * rvec[3] + rvec[1] * rvec[2]),
            1. - 2. * rvec[0] ** 2 - 2. * rvec[1] ** 2
        ), dim=0).view(-1, 3, 3)

    def forward(self, xyz, **render_kwargs):
        """ Generate sample points
        Args:
        xyz: [N,3] points in real world coordinates
        render_kwargs: other render parameters

        Return:
        xyz: [N,3] Transformed  points in real world coordinates
        """
        # Rotate ray directions w.r.t. rvec
        c2w = self.rot_mats()
        #rays_d = torch.sum(rays_d[..., None, :3] * c2w[:, :3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera w.r.t. tvec
        xyz = torch.sum(xyz[...,None, :3] * c2w[:,:3, :3],-1) + self.tvec

        return xyz