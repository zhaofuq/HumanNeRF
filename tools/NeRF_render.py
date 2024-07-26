import os
import sys
sys.path.append('..')

import cv2
import numpy as np 
import torch
from torch import nn

import glob
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import cfg
from engine import render

import torch.nn.functional as F
import torchvision.transforms as TF

from data.datasets.utils import *
from engine.util import *
from modeling import build_model, build_extractor
from solver import make_optimizer
from apex import amp

from modeling.ImageBlendingNets import ImageBlending2 as ImageBlendingNet
from layers import  warp, warp_pose
from layers.depth_warp import depth_transform, depth_transform_pose

torch.cuda.set_device(1)

training_folder = f'/sharedata/home/zhaofq/data/CVPR2022/datasets/kinect'
output_folder = f'/sharedata/home/zhaofq/data/CVPR2022/outputs/kinect/finetune_syl2'

cfg.merge_from_file(os.path.join(output_folder,'configs.yml'))
cfg.MODEL.BOARDER_WEIGHT = 1e-10
cfg.DATASETS.TRAIN = training_folder 

#model loading
model = build_model(cfg).cuda()
optimizer = make_optimizer(cfg, model)
extractor = build_extractor(cfg).cuda()

model_path = glob.glob(os.path.join(output_folder,'*.pth'))
model_iter = [int(pth.replace('.pth','').split('_')[-1]) for pth in model_path]
epoch = max(model_iter)
model.load_state_dict(torch.load(os.path.join(output_folder, 'model_epoch_%d.pth'%epoch), map_location='cpu'))   
optimizer.load_state_dict(torch.load(os.path.join(output_folder, 'optimizer_epoch_%d.pth'%epoch), map_location='cpu'))
extractor.load_state_dict(torch.load(os.path.join(output_folder, 'extractor_epoch_%d.pth'%epoch), map_location='cpu'))   
model.eval()

blendingNet = ImageBlendingNet(cfg, 13)
pretrain_model = torch.load(os.path.join('/sharedata/home/zhaofq/data/CVPR2022/outputs/RealTimeHFVV/suoxin_0719', 'model_858000.pth'), map_location='cpu')
#pretrain_model = torch.load(os.path.join(f'/sharedata/home/zhaofq/data/CVPR2022/outputs/RealTimeHFVV/data1031/', 'model_216000.pth'), map_location='cpu')
pretrain_dict = {k[10:]:v for k,v in pretrain_model['model'].items() if 'blend_net' in k }
blendingNet.load_state_dict(pretrain_dict)
blendingNet = blendingNet.cuda()

#params
scene = 'syl'
src_size = (256,256)
tgt_size = (1080,1080)
frame_id = 170
resolution = cv2.imread(os.path.join(training_folder,f'{scene}/img/{frame_id}/img_0000.jpg')).shape[0]
upsize = 1080
depth_thres = 1.0
b, h, w = 1,src_size[0], src_size[1]


camposes = np.loadtxt(os.path.join(training_folder,f'{scene}/CamPose.inf')) #_120
FTs = campose_to_extrinsic(camposes)
FKs = read_intrinsics(os.path.join(training_folder,f'{scene}/Intrinsic.inf')) #_120
FTs = torch.from_numpy(FTs).float()
FKs = torch.from_numpy(FKs).float()

camposes = np.loadtxt(os.path.join(training_folder,f'{scene}/CamPose.inf'))
Ts = campose_to_extrinsic(camposes)
Ks = read_intrinsics(os.path.join(training_folder,f'{scene}/Intrinsic.inf'))
Ts = torch.from_numpy(Ts).float()
Ks = torch.from_numpy(Ks).float()

FTs = Ts[1].unsqueeze(0).repeat(120, 1, 1)
FKs = Ks[1].unsqueeze(0).repeat(120, 1, 1)
mask_blur = SmoothConv2D(9, 'cpu')

for frame_id in tqdm(range(170,310)):
    cid = 25#(frame_id - 170 + 20) % 120
    for cam_id in range(1, 2):
        cur_depth_img, pre_depth_img, nxt_depth_img = render_view_and_depth(training_folder, model, extractor,
                                                                    FKs.clone(), FTs.clone(), Ks.clone(), Ts.clone(), 
                                                                    scene, frame_id, cam_id,
                                                                    tgt_size = tgt_size, src_size = src_size, resolution = resolution)

        pre_cam, nxt_cam = get_src_cam(FTs[cam_id], Ts)
        cur_intrinc, cur_extrinc, prev_intrinc, prev_extrinc, nxt_intrinc, nxt_extrinc = read_camera(FTs.clone(),FKs.clone(),
                                                                                                     Ts.clone(),Ks.clone(), 
                                                                                                     cam_id , pre_cam, nxt_cam, 
                                                                                                     size = src_size, resolution = resolution)

        prev_image_hw, prev_mask_hw = read_image(training_folder, scene, frame_id, pre_cam, size = (h,w))
        prev_depth_hw = TF.functional.resize(pre_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0)

        nxt_image_hw, nxt_mask_hw = read_image(training_folder, scene, frame_id, nxt_cam, size = (h,w))
        nxt_depth_hw = TF.functional.resize(nxt_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0) 

        cur_image = cur_depth_img[:,:,:3].clone()
        cur_image[cur_depth_img[:,:,3] < depth_thres] = 1.0

        cur_mask_hw = torch.ones_like(cur_image)
        cur_mask_hw[cur_depth_img[:,:,3] < depth_thres] = 0.0

        cur_image_hw = TF.functional.resize(cur_image.permute(2,0,1),(h,w)).unsqueeze(0) * 2.0 - 1.0
        cur_depth_hw = TF.functional.resize(cur_depth_img[:,:,3:].permute(2,0,1), (h,w)).unsqueeze(0)
        cur_depth_hw[cur_depth_hw < depth_thres] = 0.
        cur_image_hw[:,:,cur_depth_hw[0,0] < depth_thres] = 1.0

        prev_depth_tgt = depth_transform(prev_depth_hw, prev_intrinc, torch.matmul(cur_extrinc.inverse(), prev_extrinc))
        nxt_depth_tgt = depth_transform(nxt_depth_hw, nxt_intrinc, torch.matmul(cur_extrinc.inverse(), nxt_extrinc))

        prev_image_depth = torch.cat((prev_image_hw[:,:3], prev_depth_tgt), dim=1)
        nxt_image_depth = torch.cat((nxt_image_hw[:,:3], nxt_depth_tgt), dim=1)

        prev_image_warp = warp(cur_depth_hw, prev_image_depth, prev_intrinc, cur_intrinc, prev_extrinc, cur_extrinc)
        nxt_image_warp = warp(cur_depth_hw, nxt_image_depth, nxt_intrinc, cur_intrinc, nxt_extrinc, cur_extrinc)

        depth1_diff = torch.abs(cur_depth_hw - prev_image_warp[:,3].view((b,1,h,w)))
        depth2_diff = torch.abs(cur_depth_hw - nxt_image_warp[:,3].view((b,1,h,w)))
        occlusion = (torch.abs(depth1_diff) < torch.abs(depth2_diff)).float()

        cur_image_depth = torch.cat([cur_image_hw, occlusion], dim=1)
        #cur_image_depth = occlusion
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
        depth_scale[depth_scale < depth_thres] = 0.

        cur_intrinc_scale, cur_extrinc, prev_intrinc_scale, prev_extrinc, nxt_intrinc_scale, nxt_extrinc = read_camera(FTs.clone() ,FKs.clone() ,
                                                                                                                     Ts.clone() ,Ks.clone(), 
                                                                                                                     cam_id , pre_cam, nxt_cam, 
                                                                                                                     size = (upsize,upsize),resolution = resolution)

        prev_image_scale , prev_mask_scale = read_image(training_folder, scene, frame_id, pre_cam, size = (upsize,upsize))
        prev_depth_scale = TF.functional.resize(pre_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)

        nxt_image_scale , nxt_mask_scale = read_image(training_folder, scene, frame_id, nxt_cam, size = (upsize,upsize))
        nxt_depth_scale = TF.functional.resize(nxt_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)

        prev_depth_tgt = depth_transform(prev_depth_scale, prev_intrinc_scale, torch.matmul(cur_extrinc.inverse(), prev_extrinc))
        nxt_depth_tgt = depth_transform(nxt_depth_scale, nxt_intrinc_scale, torch.matmul(cur_extrinc.inverse(), nxt_extrinc))


        prev_image_depth_scale = torch.cat((prev_image_scale[:,:3], prev_depth_tgt), dim=1)
        nxt_image_depth_scale = torch.cat((nxt_image_scale[:,:3], nxt_depth_tgt), dim=1)

        prev_image_warp_scale = warp(depth_scale, prev_image_depth_scale, prev_intrinc_scale, cur_intrinc_scale, prev_extrinc, cur_extrinc)
        nxt_image_warp_scale = warp(depth_scale, nxt_image_depth_scale, nxt_intrinc_scale, cur_intrinc_scale, nxt_extrinc, cur_extrinc)

        cur_image_scale = TF.functional.resize(cur_depth_img[:,:,:3].permute(2,0,1),(upsize,upsize)).unsqueeze(0) * 2.0 - 1.0
        cur_mask_scale = TF.functional.resize(cur_depth_img[:,:,3:].permute(2,0,1), (upsize,upsize)).unsqueeze(0)
        cur_image_scale[:,:,cur_mask_scale[0,0] < depth_thres] = 1.0

        # depth1_diff = torch.abs(depth_scale - prev_image_warp_scale[:,3:].view((b,1,upsize,upsize)))
        # depth2_diff = torch.abs(depth_scale - nxt_image_warp_scale[:,3:].view((b,1,upsize,upsize)))

        image_scale = weight_scale[:,0,:,:].unsqueeze(1).cpu() * prev_image_warp_scale[:,:3] \
                    + weight_scale[:,1,:,:].unsqueeze(1).cpu()  * nxt_image_warp_scale[:,:3] \
                    + weight_scale[:,2,:,:].unsqueeze(1).cpu()  * cur_image_scale[:,:3] 

        image_scale[:,:,cur_mask_scale[0,0] < depth_thres] = - 1.0
        image_scale = (image_scale + 1.0 ) / 2.0 

        image_scale[image_scale>1.0] = 1.0
        image_scale[image_scale<0.0] = 0.0

        nerf = cur_depth_img[:,:,:3].detach().cpu().clone()
        nerf[nerf>1.0] = 1.0
        nerf[nerf<0.0] = 0.0
        nerf[cur_depth_img[:,:,3]<1.0,:] = 0.0 #1.0

        if not os.path.exists(os.path.join(output_folder,  f'{scene}_eval')):
            os.mkdir(os.path.join(output_folder,  f'{scene}_eval'))
        plt.imsave(os.path.join(output_folder, f'{scene}_eval/img_%04d_%02d.jpg'%(frame_id, cam_id)),nerf.numpy())

        #         if not os.path.exists(os.path.join(output_folder,  f'{scene}_nerf')):
        #             os.mkdir(os.path.join(output_folder,  f'{scene}_nerf'))
        #         plt.imsave(os.path.join(output_folder, f'{scene}_nerf/img_%04d_%02d.jpg'%(frame_id, cam_id)),nerf.numpy())

        #smooth boundary
        boundary = find_boundary(cur_mask_scale, mask_blur, size = (upsize,upsize), resolution = upsize)

        image_scale = image_scale[0].permute(1,2,0) 

        image_scale = image_scale.detach().cpu().reshape((upsize, upsize, 3))
        image_scale[boundary[0,0] > 0.5,:] = 0.0

        if not os.path.exists(os.path.join(output_folder,  f'{scene}_blender')):
            os.mkdir(os.path.join(output_folder,  f'{scene}_blender'))
        plt.imsave(os.path.join(output_folder, f'{scene}_blender/img_%04d_%02d.jpg'%(frame_id, cam_id)),image_scale.numpy())