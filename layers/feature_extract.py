import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

def bilinear_sample_function(feature_map, xyz, mask, padding_mode):
    b, c, h, w = feature_map.shape[:4]
    
    xx = xyz[:, 0, :].unsqueeze(-1)
    yy = xyz[:, 1, :].unsqueeze(-1)
    
    xx_norm = xx / (w - 1) * 2 - 1
    yy_norm = yy / (h - 1) * 2 - 1
    
    pixel_coords = torch.stack([xx_norm, yy_norm], dim=2).reshape((b, -1, 1,  2))  # [B, H*W, 2]
    feature = F.grid_sample(feature_map, pixel_coords, padding_mode=padding_mode)[:,:,:,0]
    mask_flag = F.grid_sample(mask, pixel_coords, padding_mode=padding_mode)[:,:,:,0]
    return feature, mask_flag

def projection(points, intrinc,  extrinc):
    extrinc = extrinc.squeeze(0).unsqueeze(0)
    rot = extrinc[:,:3, :3] # 3 x 3
    trans = extrinc[:,:3, 3:4]  # 3 x 1

    # trans.view(-1, 3, 1) + rot * points.transpose(1, 2)
    points = torch.baddbmm(trans.view(-1, 3, 1), rot, points.transpose(1, 2))  # [B, 3, N]
    
    # normalization z=1
    rays_dir = points[:,:3]/points[:, 2].unsqueeze(1)

    # 
    rays_dir = torch.matmul(rot.transpose(1, 2), rays_dir)
    homo = torch.matmul(intrinc, points)
    
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz, rays_dir

def depth_normlization(z_buffer, one_hot):
    if(one_hot):
        z_buffer = (z_buffer-1800.0)/1000.0
        soft_dim = 64
        z_feat = torch.zeros(z_buffer.size(0), soft_dim, z_buffer.size(2)).to(z_buffer.device) # [1, 64, 10000]
        z_norm = (z_buffer.clamp(-1, 1) + 1) / 2.0 * (soft_dim - 1)
        z_floor = torch.floor(z_norm) #[1, 1, 10000]
        z_ceil = torch.ceil(z_norm)
        z_floor_value = 1 - (z_norm - z_floor) #[1, 1, 10000]
        z_ceil_value = 1 - (z_ceil - z_norm)
        z_feat = z_feat.scatter(dim=1, index=z_floor.long(), src=z_floor_value)
        z_feat = z_feat.scatter(dim=1, index=z_ceil.long(), src=z_ceil_value)
    else:
        z_feat =   (z_buffer-1800.0)/1000.0
    return z_feat.view((z_buffer.size(0), -1, z_buffer.size(2)))

def pointsProjection(points, intrinc, extrinc, size = (256, 256), scale=1):
    b = points.shape[0]
    h, w = size[0], size[1]
    extrinc = torch.inverse(extrinc)
    intrinc_copy = intrinc.clone()
    intrinc_copy[:,:2] = intrinc_copy[:,:2]/scale
    xyz, rays_dir = projection(points, intrinc_copy, extrinc)
    xx = xyz[:, 0, :].unsqueeze(-1)
    yy = xyz[:, 1, :].unsqueeze(-1)
    xx_norm = xx / (w - 1) * 2 - 1
    yy_norm = yy / (h - 1) * 2 - 1
    pixel_coords = torch.stack([xx_norm, yy_norm], dim=2).reshape((b, -1, 1,  2))  # [B, H*W, 2]
    z_buffer = xyz[:, 2, :].unsqueeze(1)
    return pixel_coords, z_buffer

def feature_extract(feature_map, pixel_coords):
    feature = F.grid_sample(feature_map, pixel_coords, padding_mode='border')[:,:,:,0]
    return feature
