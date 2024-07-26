import os
import sys
sys.path.append('..')

import torch
import numpy as np 
import matplotlib.pyplot as plt

import glob
from tqdm import tqdm
from PIL import Image
    
from config import cfg
from engine import render
from engine import blender

from data.datasets.utils import *
from modeling import build_model

from modeling import ImageBlendingNet
from modeling.ImageBlendingNets import ImageBlending2 as ImageBlendingNet

torch.cuda.set_device(0)


def get_bbox(path):
    if path[-3:] == 'npy':
        tmp = np.load(path)
    elif path[-3:] == 'txt':
        tmp = np.loadtxt(path)
    else:
        return
    
    vs = torch.tensor(tmp[:,:3])

    max_xyz = torch.max(vs, dim=0)[0]
    min_xyz = torch.min(vs, dim=0)[0]
    tmp = (max_xyz - min_xyz) * 0.1
    max_xyz = max_xyz + tmp
    min_xyz = min_xyz - tmp

    minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
    maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
    bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
    bbox = torch.from_numpy(bbox.astype(np.float32)).reshape((8, 3))
    return bbox


def render_view_and_depth(training_folder, frame_id, cam_id, tgt_size = (512, 512), src_size = (256,256)):
    
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
    
    fid = frame_id - cfg.DATASETS.START_FRAME
    
    bboxes = []
    vertices = torch.from_numpy(np.load(os.path.join(training_folder, f'pointclouds/frame{frame_id+1}.npy')).astype(np.float32))
    bboxes.append(get_bbox(os.path.join(training_folder, f'pointclouds/frame{frame_id+1}.npy')).reshape(1,8,3))
    bboxes = torch.cat(bboxes, dim=0)
    bboxes = bboxes[0].reshape(8,3)
    
    FKs[:,:2,:] = FKs[:,:2,:] * tgt_size[0] / 1080
    Ks[:,:2,:] = Ks[:,:2,:] * src_size[0] / 1080
    
    Rts = torch.tensor(np.load(os.path.join(training_folder, f'skeletons/Rt%d.npy' % frame_id)).astype(np.float32))
    global_ts = torch.tensor(np.load(os.path.join(training_folder, f'skeletons/global_t%d.npy' % frame_id)).astype(np.float32)).squeeze(0)
    skeletons = torch.tensor(np.load(os.path.join(training_folder, f'skeletons/skeletonpose%d.npy' % frame_id)).astype(np.float32)).squeeze(0)
    
    stage2, stage1 = render(model, FKs[cam_id], FTs[cam_id], tgt_size, Rts, global_ts, skeletons, vertices[:,:3], bboxes = bboxes, density_thres=0.0)
    cur_img = stage2[0].detach().cpu()
    cur_img = (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
    cur_depth = stage2[1].detach().cpu()
    cur_depth_img = torch.cat([cur_img, cur_depth.unsqueeze(-1)], dim=-1)
    
    pre_cam = cam_id // skip
    nxt_cam = ( cam_id // skip + 1) % 6
    
    stage2, stage1 = render(model, Ks[pre_cam], Ts[pre_cam], src_size, Rts, global_ts, skeletons, vertices[:,:3], bboxes = bboxes,  density_thres=0.0)
    pre_img = stage2[0].detach().cpu()
    pre_depth = stage2[1].detach().cpu()
    pre_depth_img = torch.cat([pre_img, pre_depth.unsqueeze(-1)], dim=-1)

    stage2, stage1 = render(model, Ks[nxt_cam], Ts[nxt_cam], src_size, Rts, global_ts, skeletons, vertices[:,:3],  bboxes = bboxes, density_thres=0.0)
    nxt_img = stage2[0].detach().cpu()
    nxt_depth = stage2[1].detach().cpu()
    nxt_depth_img = torch.cat([nxt_img, nxt_depth.unsqueeze(-1)], dim=-1)
    
    return cur_depth_img, pre_depth_img, nxt_depth_img


training_folder = f'/sharedata/home/zhaofq/data/datasets/kinect/spiderman1/'
output_folder = f'/sharedata/home/zhaofq/data/outputs/kinect/spiderman1_lambda/'
save_name = 'fvv'

cfg.merge_from_file(os.path.join(output_folder,'configs.yml'))
cfg.MODEL.BOARDER_WEIGHT = 1e-10
cfg.DATASETS.TRAIN = training_folder 

src_size = (256,256)
tgt_size = (720,720)
skip = 20

#NeRF Model
model = build_model(cfg).cuda()

model_path = glob.glob(os.path.join(output_folder,'*.pth'))
model_iter = [int(pth.replace('.pth','').split('_')[-1]) for pth in model_path]
epoch = max(model_iter)
model.load_state_dict(torch.load(os.path.join(output_folder, 'model_epoch_%d.pth'%epoch), map_location='cpu'))                  
model.eval()

#human fvv model
blendingNet = ImageBlendingNet(cfg, 13)
pretrain_model = torch.load(os.path.join(f'/sharedata/home/zhaofq/data/outputs/RealTimeHFVV/suoxin_0719/', 'model_792000.pth'), map_location='cpu')
pretrain_dict = {k[10:]:v for k,v in pretrain_model['model'].items() if 'blend_net' in k }
blendingNet.load_state_dict(pretrain_dict)
blendingNet = blendingNet.cuda()

#render
for frame_id in tqdm(range(1280, 1281)):

    for cam_id in tqdm(range(0,120)):
    
        cur_depth_img, pre_depth_img, nxt_depth_img = render_view_and_depth(training_folder, frame_id, cam_id, tgt_size = tgt_size, src_size=src_size)
        
        image_scale, weight_scale, image_pred, weight = blender(blendingNet, training_folder, cur_depth_img, pre_depth_img ,nxt_depth_img, 
                                                                frame_id, cam_id , upsize = 720)
        
        nerf = cur_depth_img[:,:,:3].detach().cpu().clone()
        nerf[nerf>1.0] = 1.0
        nerf[nerf<0.0] = 0.0
        nerf[cur_depth_img[:,:,3]<1.0,:] = 1.0
        if not os.path.exists(os.path.join(output_folder,  f'nerf_{save_name}')):
            os.mkdir(os.path.join(output_folder,  f'nerf_{save_name}'))
        plt.imsave(os.path.join(output_folder, f'nerf_{save_name}/img_%04d_%02d.jpg'%(frame_id, cam_id)),nerf.numpy())

        if not os.path.exists(os.path.join(output_folder,  f'blender_{save_name}')):
            os.mkdir(os.path.join(output_folder,  f'blender_{save_name}'))
        plt.imsave(os.path.join(output_folder, f'blender_{save_name}/img_%04d_%02d.jpg'%(frame_id, cam_id)),image_scale.numpy())