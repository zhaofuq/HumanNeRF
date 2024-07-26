import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import knn_points

def body_weights_sample(weights, vertices, xyz):

    ret = knn_points(xyz.unsqueeze(0), vertices.unsqueeze(0), None, None, 1)
    dist, idx = ret[0].squeeze(-1) ,ret[1].squeeze(-1)
    weight = weights[idx]

    return weight

def depth_transform(depth, intrinc ,RT):
    
    b, c, h, w = depth.shape
    x = torch.arange(0, w, device=depth.device).float()
    y = torch.arange(0, h, device=depth.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth.reshape((b, h * w, 1))
    
    center = torch.cat((intrinc[:, 0, 2].unsqueeze(1), intrinc[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc[:, 0, 0].unsqueeze(1), intrinc[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz = (xyz * dd).transpose(1, 2)
    
    R12 = RT[:,:3,:3]
    t12 = RT[:,:3,3]
    xyz = torch.matmul(R12, xyz) + t12.reshape((-1,3,1))
#     xy = xyz[:,:2]/xyz[:,2].reshape((b, 1, -1))
    
    dd_transform = xyz[:,2].reshape((b,1,h, w))
    # dd_transform
    
    return dd_transform#, xyz

def depth_transform_pose(depth, intrinc , cur_extrinc, prev_extrinc,
                        Rts_src,  Rts_tgt, global_ts_src, global_ts_tgt, skinning_weights, vertices):
    
    b, c, h, w = depth.shape
    x = torch.arange(0, w, device=depth.device).float()
    y = torch.arange(0, h, device=depth.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth.reshape((b, h * w, 1))
    mask = dd > 0.5
    
    center = torch.cat((intrinc[:, 0, 2].unsqueeze(1), intrinc[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc[:, 0, 0].unsqueeze(1), intrinc[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz_mask = (xyz[mask.squeeze(-1),:] * dd[mask.squeeze(-1),:]).unsqueeze(0).transpose(1, 2)
    
    #cam2world
    RT =  prev_extrinc
    R12 = RT[:,:3,:3]
    t12 = RT[:,:3,3]
    xyz_mask = torch.matmul(R12, xyz_mask) + t12.reshape((-1,3,1))
    xyz_mask = xyz_mask[0].permute(1,0)
  
    #cur pose->novel pose
    W = body_weights_sample(skinning_weights, vertices, xyz_mask)
    wRts_tgt = torch.matmul(W, Rts_tgt).reshape(-1, 4, 4)
    wRts_src = torch.matmul(W, Rts_src).reshape(-1, 4, 4)
    wRts = torch.matmul(wRts_tgt, wRts_src.inverse())

    xyz_mask = torch.sum((xyz_mask - global_ts_src)[:,None,:] * wRts[:,:3,:3], dim=-1) + wRts[:,:3,3] + global_ts_tgt

    #world2cam
    RT = cur_extrinc.inverse()
    R12 = RT[:,:3,:3]
    t12 = RT[:,:3,3]
    xyz_mask = torch.matmul(R12, xyz_mask.unsqueeze(0).transpose(1,2)) + t12.reshape((-1,3,1))

    dd_transform = torch.zeros_like(depth)
    mask = mask.reshape(b,c,h,w)
    dd_transform[mask] = xyz_mask[:,2]

    return dd_transform
    
