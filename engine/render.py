import torch

from layers import intersection
from utils import batchify_ray, ray_sampling
import numpy as np
import os
import torch


'''
Sample rays from views (and images) with/without masks

--------------------------
INPUT Tensors
K: intrinsics of camera (3,3)
T: extrinsic of camera (4,4)
image_size: the size of image [H,W]

ROI:  2D ROI bboxes  (4) left up corner(x,y) followed the height and width  (h,w)

masks:(M,H,W)
-------------------
OUPUT:
list of rays:  (N,6)  dirs(3) + pos(3)
RGB:  (N,C)
'''


def render(model, feature_maps, K, T, img_size, Rts, global_ts, skeletons,  vertices , scene, ROI = None, bboxes = None,only_coarse = False , near_fars=None):
    model.eval()
    assert not (bboxes is None and near_far is None), 'either bbox or near_far should not be None.'
    mask = torch.ones(img_size[0],img_size[1])
    if ROI is not None:
        mask = torch.zeros(img_size[0],img_size[1])
        mask[ROI[0]:ROI[0]+ROI[2], ROI[1]:ROI[1]+ROI[3]] = 1.0
    rays,_ = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), img_size, masks=mask.unsqueeze(0))

    #frame_ids = torch.ones((rays.size(0),1)) * frame_id
    #rays = torch.cat([rays,frame_ids],dim=1)
    
    if Rts is not None:
        Rts = Rts.unsqueeze(0).repeat(rays.size(0),1,1)
        Rts = Rts.cuda()
        
    if global_ts is not None:
        global_ts = global_ts.unsqueeze(0).repeat(rays.size(0),1)
        global_ts = global_ts.cuda()
        
    if global_ts is not None:
        skeletons = skeletons.unsqueeze(0).repeat(rays.size(0),1,1)
        skeletons = skeletons.cuda()
        
    if bboxes is not None:

        bboxes = bboxes.unsqueeze(0).repeat(rays.size(0),1 ,1)
        bboxes = bboxes.cuda()
    
    if near_fars is not None:
        near_fars = near_fars.unsqueeze(0).repeat(rays.size(0),1)
        near_fars = near_fars.cuda()
    else:
        assert vertices is not None, 'requires pointclouds as input'
        inv_Ts = torch.inverse(T).unsqueeze(0)  #(1,4,4)
        vs = vertices.unsqueeze(-1)   #(N,3,1)
        vs = torch.cat([vs,torch.ones(vs.size(0),1,vs.size(2)) ],dim=1) #(N,4,1)

        pts = torch.matmul(inv_Ts,vs) #(N,4,1)

        pts_max = torch.max(pts, dim=0)[0].squeeze() #(4)
        pts_min = torch.min(pts, dim=0)[0].squeeze() #(4)

        pts_max = pts_max[2:3]   #(M)
        pts_min = pts_min[2:3]   #(M)
        
        near = pts_min *0.5
        near[near<(pts_max*0.1)] = pts_max[near<(pts_max*0.1)]*0.1
        far = pts_max *2

        near_fars = torch.cat([near.unsqueeze(-1), far.unsqueeze(-1)],dim=1)
        near_fars = near_fars.repeat(rays.size(0),1)
        near_fars = near_fars.cuda()

    rays = rays.cuda()
    vertices = vertices.cuda()
    
    
    with torch.no_grad():
        stage2, stage1, _ = batchify_ray(model, rays, bboxes , feature_maps, Rts, global_ts,  skeletons, vertices, scene, near_far = near_fars)
        
    rgb = torch.zeros(img_size[0],img_size[1], 3, device = stage2[0].device)
    rgb[mask>0.5,:] = stage2[0]

    depth = torch.zeros(img_size[0],img_size[1],1, device = stage2[1].device)
    depth[mask>0.5,:] = stage2[1]

    alpha = torch.zeros(img_size[0],img_size[1],1, device = stage2[2].device)
    alpha[mask>0.5,:] = stage2[2]
    
    stage2_final = [None]*3
    stage2_final[0] = rgb.reshape(img_size[0],img_size[1], 3)
    stage2_final[1] = depth.reshape(img_size[0],img_size[1])
    stage2_final[2] = alpha.reshape(img_size[0],img_size[1])

    
    rgb = torch.zeros(img_size[0],img_size[1], 3, device = stage1[0].device)
    rgb[mask>0.5,:] = stage1[0]

    depth = torch.zeros(img_size[0],img_size[1],1, device = stage1[1].device)
    depth[mask>0.5,:] = stage1[1]

    alpha = torch.zeros(img_size[0],img_size[1],1, device = stage1[2].device)
    alpha[mask>0.5,:] = stage1[2]

    stage1_final = [None]*3
    stage1_final[0] = rgb.reshape(img_size[0],img_size[1], 3)
    stage1_final[1] = depth.reshape(img_size[0],img_size[1])
    stage1_final[2] = alpha.reshape(img_size[0],img_size[1])
  
        
    return stage2_final, stage1_final
