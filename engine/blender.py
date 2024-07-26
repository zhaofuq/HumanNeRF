import os
import cv2
import numpy as np 
from PIL import Image

import torch
from torch import nn

from data.datasets.utils import *
from layers import  warp, warp_pose
from layers.depth_warp import depth_transform, depth_transform_pose

import torch.nn.functional as F
import torchvision.transforms as TF

def one_hote_depth(depth_diff):
    soft_dim = 3
    z_feat = torch.zeros(depth_diff.size(0), soft_dim, depth_diff.size(2), depth_diff.size(3)).to(depth_diff.device) # [1, 8, h, w]
    z_norm = (depth_diff.clamp(-1, 1) + 1) / 2.0 * (soft_dim - 1)
    z_floor = torch.floor(z_norm) #[1, 1, 10000]
    z_ceil = torch.ceil(z_norm)
    z_floor_value = 1 - (z_norm - z_floor) #[1, 1, 10000]
    z_ceil_value = 1 - (z_ceil - z_norm)
    z_feat = z_feat.scatter(dim=1, index=z_floor.long(), src=z_floor_value)
    z_feat = z_feat.scatter(dim=1, index=z_ceil.long(), src=z_ceil_value)

    return z_feat.view((depth_diff.size(0), soft_dim, depth_diff.size(2), depth_diff.size(3)))

def read_image(training_folder, frame_id, cam_id, size=(256,256)):
    
    image = Image.open(os.path.join(training_folder, f'img/{frame_id}/img_{cam_id:04d}.jpg'))
    image = TF.functional.to_tensor(image)
    mask = Image.open(os.path.join(training_folder, f'img/{frame_id}/mask_{cam_id:04d}.jpg'))
    mask = TF.functional.to_tensor(mask)
    image[mask<0.5] = 1.0
    image = TF.functional.resize(image,size).unsqueeze(0) * 2.0 - 1.0
    mask = TF.functional.resize(mask,size).unsqueeze(0)
    
    return image, mask

def read_camera(training_folder, cam_id , pre_cam,  nxt_cam, size = (256,256)):
    
    camposes = np.loadtxt(os.path.join(training_folder,'CamPose_120.inf'))
    FTs = campose_to_extrinsic(camposes)
    FKs = read_intrinsics(os.path.join(training_folder,'Intrinsic_120.inf'))
    FTs = torch.from_numpy(FTs).float()
    FKs = torch.from_numpy(FKs).float()

    camposes = np.loadtxt(os.path.join(training_folder,'CamPose.inf'))
    Ts = campose_to_extrinsic(camposes)
    Ks = read_intrinsics(os.path.join(training_folder,'Intrinsic.inf'))
    Ts = torch.from_numpy(Ts).float()
    Ks = torch.from_numpy(Ks).float()
    
    h,w = size[0], size[1]

    prev_extrinc = Ts[pre_cam].unsqueeze(0)
    nxt_extrinc = Ts[nxt_cam].unsqueeze(0)
    cur_extrinc = FTs[cam_id].unsqueeze(0)

    prev_intrinc = Ks[pre_cam].unsqueeze(0)
    prev_intrinc[:,:2,:] = prev_intrinc[:,:2,:] * h / 1080

    nxt_intrinc = Ks[nxt_cam].unsqueeze(0)
    nxt_intrinc[:,:2,:] = nxt_intrinc[:,:2,:] * h / 1080

    cur_intrinc = FKs[cam_id].unsqueeze(0)
    cur_intrinc[:,:2,:] = cur_intrinc[:,:2,:] * h / 1080

    return cur_intrinc, cur_extrinc, prev_intrinc, prev_extrinc, nxt_intrinc, nxt_extrinc

class SmoothConv2D(nn.Module):
    def __init__(self, kernel_size=5, device = 'cuda:0'):
        super().__init__()
        self.padding = (kernel_size - 1) // 2 

        weight = (torch.ones(
            (1, 1, kernel_size, kernel_size), 
            dtype=torch.float32
        ) / (kernel_size**2)).to(device)
        self.register_buffer('weight', weight)
        
    def forward(self, input):
        return F.conv2d(input, self.weight, padding=self.padding)

def find_boundary(coarse_mask, mask_blur_func, size = (512,512)):
    
    coarse_mask_scale = F.interpolate(coarse_mask, scale_factor = size[0]/720, mode = 'bilinear') 
    coarse_mask_scale_blur = mask_blur_func(coarse_mask_scale)
    coarse_mask_scale_blur[coarse_mask_scale_blur<1.0] = 0
    coarse_mask_scale0 = coarse_mask_scale.clone()
    coarse_mask_scale0[coarse_mask_scale0 > 0.5]= 1.0
    boundary = coarse_mask_scale0 - coarse_mask_scale_blur
    
    return boundary


def blender(blendingNet, training_folder, cur_depth_img, pre_depth_img ,nxt_depth_img, frame_id, cam_id, upsize = 720,  skip = 20):
    
    src_size = (pre_depth_img.shape[0], pre_depth_img.shape[1])
    tgt_size = (cur_depth_img.shape[0], cur_depth_img.shape[1])
    
    b, h, w = 1, src_size[0], src_size[1]

    pre_cam = cam_id // skip
    nxt_cam = (cam_id // skip + 1) % 6
    
    mask_blur = SmoothConv2D(9, 'cpu')

    cur_intrinc, cur_extrinc, prev_intrinc, prev_extrinc, nxt_intrinc, nxt_extrinc = read_camera(training_folder, cam_id , pre_cam, nxt_cam, size = src_size)

    prev_image_hw, prev_mask_hw = read_image(training_folder, frame_id, pre_cam, size = (h,w))
    prev_depth_hw = TF.functional.resize(pre_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0)

    nxt_image_hw, nxt_mask_hw = read_image(training_folder, frame_id, nxt_cam, size = (h,w))
    nxt_depth_hw = TF.functional.resize(nxt_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0) 

    cur_image = cur_depth_img[:,:,:3].clone()
    cur_image[cur_depth_img[:,:,3] < 1.0] = 1.0

    cur_mask_hw = torch.ones_like(cur_image)
    cur_mask_hw[cur_depth_img[:,:,3] < 1.0] = 0.0

    cur_image_hw = TF.functional.resize(cur_image.permute(2,0,1),(h,w)).unsqueeze(0) * 2.0 - 1.0
    cur_depth_hw = TF.functional.resize(cur_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0)
    cur_depth_hw[cur_depth_hw < 1.0] = 0.
    cur_image_hw[:,:,cur_depth_hw[0,0] < 1.0] = 1.0

    prev_depth_tgt = depth_transform(prev_depth_hw, prev_intrinc, torch.matmul(cur_extrinc.inverse(), prev_extrinc))
    nxt_depth_tgt = depth_transform(nxt_depth_hw, nxt_intrinc, torch.matmul(cur_extrinc.inverse(), nxt_extrinc))

    prev_image_depth = torch.cat((prev_image_hw[:,:3], prev_depth_tgt), dim=1)
    nxt_image_depth = torch.cat((nxt_image_hw[:,:3], nxt_depth_tgt), dim=1)

    prev_image_warp = warp(cur_depth_hw, prev_image_depth, prev_intrinc, cur_intrinc, prev_extrinc, cur_extrinc)
    nxt_image_warp = warp(cur_depth_hw, nxt_image_depth, nxt_intrinc, cur_intrinc, nxt_extrinc, cur_extrinc)

    depth1_diff = torch.abs(cur_depth_hw - prev_image_warp[:,3].view((b,1,h,w)))
    depth2_diff = torch.abs(cur_depth_hw - nxt_image_warp[:,3].view((b,1,h,w)))
    occlusion = (torch.abs(depth1_diff) < torch.abs(depth2_diff)).float()

    # depth1_diff = (depth1_diff - depth1_diff.min()) / (depth1_diff.max() - depth1_diff.min())
    # depth2_diff = (depth2_diff - depth2_diff.min()) / (depth2_diff.max() - depth2_diff.min())

    cur_image_depth = torch.cat([cur_image_hw, occlusion], dim=1)

    depth1_diff = one_hote_depth(depth1_diff)
    depth2_diff = one_hote_depth(depth2_diff)

    prev_image_depth_warp = torch.cat([prev_image_warp[:, :3], (depth1_diff)], dim=1)
    nxt_image_depth_warp = torch.cat([nxt_image_warp[:, :3], (depth2_diff)], dim=1)

    #forward
    image_pred, weight = blendingNet(prev_image_depth_warp.cuda(), nxt_image_depth_warp.cuda(), cur_image_depth.cuda())
    image_pred = (image_pred+1.0)/2.0 

    #upsampling
    weight_scale = F.interpolate(weight, scale_factor = upsize/h, mode = 'bilinear')
    depth_scale  = F.interpolate(cur_depth_img[:,:,3:].permute(2,0,1).unsqueeze(0), scale_factor = upsize/tgt_size[0], mode = 'bilinear')
    depth_scale[depth_scale<1.0] = 0.

    cur_intrinc_scale, cur_extrinc, prev_intrinc_scale, prev_extrinc, nxt_intrinc_scale, nxt_extrinc = read_camera(training_folder, cam_id , pre_cam, nxt_cam, size = (upsize,upsize))

    prev_image_scale , prev_mask_scale = read_image(training_folder, frame_id, pre_cam, size = (upsize,upsize))
    prev_depth_scale = TF.functional.resize(pre_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)

    nxt_image_scale , nxt_mask_scale = read_image(training_folder, frame_id, nxt_cam, size = (upsize,upsize))
    nxt_depth_scale = TF.functional.resize(nxt_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)

    prev_depth_tgt = depth_transform(prev_depth_scale, prev_intrinc_scale, torch.matmul(cur_extrinc.inverse(), prev_extrinc))
    nxt_depth_tgt = depth_transform(nxt_depth_scale, nxt_intrinc_scale, torch.matmul(cur_extrinc.inverse(), nxt_extrinc))


    prev_image_depth_scale = torch.cat((prev_image_scale[:,:3], prev_depth_tgt), dim=1)
    nxt_image_depth_scale = torch.cat((nxt_image_scale[:,:3], nxt_depth_tgt), dim=1)

    prev_image_warp_scale = warp(depth_scale, prev_image_depth_scale, prev_intrinc_scale, cur_intrinc_scale, prev_extrinc, cur_extrinc)
    nxt_image_warp_scale = warp(depth_scale, nxt_image_depth_scale, nxt_intrinc_scale, cur_intrinc_scale, nxt_extrinc, cur_extrinc)

    cur_image_scale = TF.functional.resize(cur_depth_img[:,:,:3].permute(2,0,1),(upsize,upsize)).unsqueeze(0) * 2.0 - 1.0
    cur_mask_scale = TF.functional.resize(cur_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)
    cur_image_scale[:,:,cur_mask_scale[0,0] < 1.0] = 1.0

    depth1_diff = torch.abs(depth_scale - prev_image_warp_scale[:,3:].view((b,1,upsize,upsize)))
    depth2_diff = torch.abs(depth_scale - nxt_image_warp_scale[:,3:].view((b,1,upsize,upsize)))

    image_scale = weight_scale[:,0,:,:].unsqueeze(1).cpu() * prev_image_warp_scale[:,:3] \
                + weight_scale[:,1,:,:].unsqueeze(1).cpu()  * nxt_image_warp_scale[:,:3] \
                + weight_scale[:,2,:,:].unsqueeze(1).cpu()  * cur_image_scale[:,:3] 

    image_scale[:,:,cur_mask_scale[0,0] < 1.0] = 1.0
    image_scale = (image_scale + 1.0 ) / 2.0 
    
    boundary = find_boundary(cur_mask_scale, mask_blur, size = (upsize,upsize))

    image_scale = image_scale[0].permute(1,2,0)
    image_scale = image_scale.detach().cpu().reshape((upsize, upsize, 3))
    image_scale[boundary[0,0] > 0.5,:] = 1.0
    
    return image_scale, weight_scale, image_pred, weight