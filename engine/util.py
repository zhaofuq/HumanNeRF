import os
import torch
import cv2
import numpy as np 
from torch import nn
from PIL import Image
from engine import render
import torch.nn.functional as F
import torchvision.transforms as TF

def get_src_cam(T, Ts):
    
    value, indices = torch.topk(torch.norm(T[None,:3,3] - Ts[:,:3,3], dim=-1), largest = False, k=2)
    
    if indices[0] < indices[1]:
        pre_cam = indices[0] 
        nxt_cam = indices[1]
    else:
        pre_cam = indices[1] 
        nxt_cam = indices[0] 
        
    if pre_cam ==0 and nxt_cam == 5:
        pre_cam = 5
        nxt_cam=0

    return pre_cam, nxt_cam

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
    tmp = (max_xyz - min_xyz) * 0.2
    max_xyz = max_xyz + tmp
    min_xyz = min_xyz - tmp

    minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
    maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
    bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
    bbox = torch.from_numpy(bbox.astype(np.float32)).reshape((8, 3))
    return bbox


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

def read_image(training_folder, scene, frame_id, cam_id, size=(256,256)):
    
    image = Image.open(os.path.join(training_folder, f'{scene}/img/{frame_id}/img_{cam_id:04d}.jpg'))
    image = TF.functional.to_tensor(image)
    mask = Image.open(os.path.join(training_folder, f'{scene}/img/{frame_id}/mask_{cam_id:04d}.jpg'))
    mask = TF.functional.to_tensor(mask)
    image[mask<0.5] = 1.0
    image = TF.functional.resize(image,size).unsqueeze(0) * 2.0 - 1.0
    mask = TF.functional.resize(mask,size).unsqueeze(0)
    
    return image, mask


def read_camera(FTs, FKs, Ts ,Ks, cam_id , pre_cam,  nxt_cam, size = (256,256), resolution = 1080):

    h,w = size[0], size[1]

    prev_extrinc = Ts[pre_cam].unsqueeze(0)
    nxt_extrinc = Ts[nxt_cam].unsqueeze(0)
    cur_extrinc = FTs[cam_id].unsqueeze(0)

    prev_intrinc = Ks[pre_cam].unsqueeze(0)
    prev_intrinc[:,:2,:] = prev_intrinc[:,:2,:] * h / resolution

    nxt_intrinc = Ks[nxt_cam].unsqueeze(0)
    nxt_intrinc[:,:2,:] = nxt_intrinc[:,:2,:] * h / resolution

    cur_intrinc = FKs[cam_id].unsqueeze(0)
    cur_intrinc[:,:2,:] = cur_intrinc[:,:2,:] * h / resolution

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

def find_boundary(coarse_mask, mask_blur_func, size = (512,512), resolution = 1080):
    
    coarse_mask_scale = F.interpolate(coarse_mask, scale_factor = size[0]/resolution, mode = 'bilinear') 
    coarse_mask_scale_blur = mask_blur_func(coarse_mask_scale)
    coarse_mask_scale_blur[coarse_mask_scale_blur<1.0] = 0
    coarse_mask_scale0 = coarse_mask_scale.clone()
    coarse_mask_scale0[coarse_mask_scale0 > 0.5]= 1.0
    boundary = coarse_mask_scale0 - coarse_mask_scale_blur
    
    return boundary

def render_view_and_depth(training_folder, model, extractor, FKs, FTs, Ks, Ts, scene, frame_id, cam_id, tgt_size = (512, 512),src_size = (256,256), resolution = 1080):
    
    bboxes = []
    vertices = torch.from_numpy(np.load(os.path.join(training_folder, f'{scene}/pointclouds/frame{frame_id+1}.npy')).astype(np.float32))
    bboxes.append(get_bbox(os.path.join(training_folder, f'{scene}/pointclouds/frame{frame_id+1}.npy')).reshape(1,8,3))
    bboxes = torch.cat(bboxes, dim=0)
    bboxes = bboxes[0].reshape(8,3)

    Rts = torch.tensor(np.load(os.path.join(training_folder, f'{scene}/skeletons/Rt%d.npy' % frame_id)).astype(np.float32))
    global_ts = torch.tensor(np.load(os.path.join(training_folder, f'{scene}/skeletons/global_t%d.npy' % frame_id)).astype(np.float32)).squeeze(0)
    skeletons = torch.tensor(np.load(os.path.join(training_folder, f'{scene}/skeletons/skeletonpose%d.npy' % frame_id)).astype(np.float32)).squeeze(0)
    
    imgs = []
    # nearest = torch.argmin(torch.norm(FTs[cam_id,:3,3] - Ts[:,:3,3], dim=-1))
    # cams = [(Ts.size(0) - 1  + nearest) % Ts.size(0), nearest, (Ts.size(0) + 1 + nearest) % Ts.size(0)]
    cams = [0, 1, 2, 3, 4, 5]
    
    for i in cams:
        img = Image.open(os.path.join(training_folder ,f'{scene}/img/{frame_id}/img_{i:04d}.jpg'))
        img = TF.functional.to_tensor(img)
        img = TF.functional.resize(img, (256,256))

        mask = Image.open(os.path.join(training_folder,f'{scene}/img/{frame_id}/mask_{i:04d}.jpg'))
        mask = TF.functional.to_tensor(mask)
        mask = TF.functional.resize(mask[:1,:,:], (256,256))
        img = torch.cat([img, mask], dim=0)
        imgs.append(img.unsqueeze(0))
        
    imgs = torch.cat(imgs, dim=0)
    imgs = 2. * imgs.cuda() - 1.0
    feature_maps = extractor(imgs)

    mKs = torch.index_select(Ks, 0, torch.tensor(cams))
    mTs = torch.index_select(Ts, 0, torch.tensor(cams))
    mKs[:,:2,:] = mKs[:,:2,:] * 256 / resolution
    scene_dict = {'scene': scene,'Ks':mKs.cuda(),'Ts':mTs.cuda()}
   
    FKs[:,:2,:] = FKs[:,:2,:] * tgt_size[0] / resolution
    Ks[:,:2,:] = Ks[:,:2,:] * src_size[0] / resolution

    stage2, stage1 = render(model, feature_maps, FKs[cam_id], FTs[cam_id], tgt_size, Rts, global_ts, skeletons, vertices[:,:3], scene_dict, bboxes = bboxes)
    cur_img = stage2[0].detach().cpu()
    cur_img = (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
    cur_depth = stage2[1].detach().cpu()
    cur_depth_img = torch.cat([cur_img, cur_depth.unsqueeze(-1)], dim=-1)
    
    #pre_cam = cam_id // skip
    pre_cam, nxt_cam = get_src_cam(FTs[cam_id], Ts)
    stage2, stage1 = render(model, feature_maps, Ks[pre_cam], Ts[pre_cam], src_size, Rts, global_ts, skeletons, vertices[:,:3], scene_dict,  bboxes = bboxes)
    pre_img = stage2[0].detach().cpu()
    pre_depth = stage2[1].detach().cpu()
    pre_depth_img = torch.cat([pre_img, pre_depth.unsqueeze(-1)], dim=-1)
    
    #nxt_cam = ( cam_id // skip + 1) % 6
    stage2, stage1 = render(model, feature_maps,  Ks[nxt_cam], Ts[nxt_cam], src_size, Rts, global_ts, skeletons, vertices[:,:3], scene_dict, bboxes = bboxes)
    nxt_img = stage2[0].detach().cpu()
    nxt_depth = stage2[1].detach().cpu()
    nxt_depth_img = torch.cat([nxt_img, nxt_depth.unsqueeze(-1)], dim=-1)
    
    return cur_depth_img, pre_depth_img, nxt_depth_img

def blender(blendingNet, training_folder, FTs, FKs, Ts, Ks, cam_id, src_size = (256,256), upsize = 720):

    b, h, w = 1, src_size[0], src_size[1]
    pre_cam, nxt_cam = get_src_cam(FTs[cam_id], Ts)
    
    mask_blur = SmoothConv2D(9, 'cpu')

    cur_intrinc, cur_extrinc, prev_intrinc, prev_extrinc, nxt_intrinc, nxt_extrinc = read_camera(FTs.clone(),FKs.clone(),
                                                                                                 Ts.clone(),Ks.clone(), 
                                                                                                 cam_id , 
                                                                                                 pre_cam, 
                                                                                                 nxt_cam, 
                                                                                                 size = src_size)

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

    cur_intrinc_scale, cur_extrinc, prev_intrinc_scale, prev_extrinc, nxt_intrinc_scale, nxt_extrinc = read_camera(FTs.clone() ,FKs.clone() ,
                                                                                                                 Ts.clone() ,Ks.clone(), 
                                                                                                                 cam_id , 
                                                                                                                 pre_cam, 
                                                                                                                 nxt_cam, 
                                                                                                                 size = (upsize,upsize))

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