
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch import nn

from utils import Trigonometric_kernel, sample_pdf , SMPLBody
from layers.RaySamplePoint import RaySamplePoint, RaySamplePoint_Near_Far, RaySamplePoint_Depth
# from .spacenet import SpaceNet
from .spacenet_v2 import SpaceNet

from .deltanet import DeltaNet
#from .deltanet_latent import DeltaNet

#from .blendnet import AttenFeatNet
from .AttenNet import AttenFeatNet

from layers.render_layer import VolumeRenderer
import time

class RFRender(nn.Module):
    
    def __init__(self, coarse_ray_sample, fine_ray_sample, boarder_weight, sample_method = 'DEPTH', same_space_net = False, TriKernel_include_input = True, cfg=None):
        super(RFRender, self).__init__()

        self.coarse_ray_sample = coarse_ray_sample
        self.fine_ray_sample = fine_ray_sample
        self.depth_ray_sample = coarse_ray_sample
        self.sample_method = sample_method
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_dir = cfg.MODEL.USE_DIR
        self.use_deform = cfg.MODEL.USE_DEFORM
        self.use_appearance = cfg.MODEL.USE_APPEARANCE
      
        self.use_identity = cfg.MODEL.USE_IDENTITY
            
        if self.sample_method == 'NEAR_FAR':
            self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)   # use near far to sample points on rays
            
        elif self.sample_method == 'DEPTH':
            self.rsp_coarse = RaySamplePoint_Depth(self.coarse_ray_sample)      # use depth to sample points on rays
        
        else:
            self.rsp_coarse = RaySamplePoint(self.coarse_ray_sample)            # use bounding box to define point sampling ranges on rays
        
        #----------------------------------------------------------------DEFORMATION-----------------------------------------------------------------#
        if self.use_deform:
            self.canonical_Rts = torch.from_numpy(np.load(os.path.join(cfg.DATASETS.TRAIN,'Canonical_Rt.npy')).astype(np.float32)).cuda() #(24, 16)
            if self.use_identity:
                Rt_identity = torch.eye(4).cuda()
                self.Rt_identity_flatten = Rt_identity.reshape(1,16)
                self.canonical_Rts = torch.cat([self.canonical_Rts, self.Rt_identity_flatten],dim=0) # (25,16)

            self.smplBody = SMPLBody(cfg.DATASETS.TRAIN)
            self.deformnet = DeltaNet(feat_dim = cfg.MODEL.FEATURE_DIM+3+1, include_input = TriKernel_include_input)
            #self.blendnet = AttenFeatNet(c_angle = 6, c_rdir = 3, feat_dim = cfg.MODEL.FEATURE_DIM, include_input = TriKernel_include_input)
            self.attenFeatNet = AttenFeatNet(c_angle = 6, c_rdir = 3, feat_dim = cfg.MODEL.FEATURE_DIM)
        #----------------------------------------------------------------DEFORMATION-----------------------------------------------------------------#


        #----------------------------------------------------------------NeRF Model-----------------------------------------------------------------#
        self.spacenet = SpaceNet(include_input = TriKernel_include_input, use_dir = self.use_dir, feature_dim = cfg.MODEL.FEATURE_DIM+3+1)
        if same_space_net:
            self.spacenet_fine = self.spacenet
        else:
            self.spacenet_fine = SpaceNet(include_input = TriKernel_include_input, use_dir = self.use_dir, feature_dim = cfg.MODEL.FEATURE_DIM+3+1)

        self.volume_render = VolumeRenderer(boarder_weight = boarder_weight)
        #----------------------------------------------------------------NeRF Model-----------------------------------------------------------------#

        #train rays proxy depth
        self.sample_depth  = None 
        self.maxs = None
        self.mins = None
        self.scale = None

        self.resolution = 1080

    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    
    def forward(self, rays, bboxes, feature_maps,  Rts = None, global_ts = None, skeletons = None, vertices = None, scene = None, only_coarse = False, sample_depth = None , near_far=None, scale = None):
    
        ray_mask = None
        ret_final = {}
        ret_final_0 = {}
        ray_marching = True

        if self.sample_method == 'NEAR_FAR':
            #training sample points useing near far mode
            assert near_far is not None, 'require near_far as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays[:,:6] , near_far = near_far)
            rays_t = rays

        elif self.sample_method == 'DEPTH':
            self.sample_depth = sample_depth
            
            if scale is not None:
                #training sample points useing depth mode
                self.scale = scale
                sampled_rays_coarse_t, sampled_rays_coarse_xyz = self.rsp_coarse.forward(rays[:,:6], depth= sample_depth, near_far=near_far, scale=self.scale)
            else:
                #validation sample points useing near far mode
                sampled_rays_coarse_t, sampled_rays_coarse_xyz = self.rsp_coarse.forward(rays[:,:6], depth= sample_depth, near_far=near_far)
            rays_t = rays
        else:
            #training sample points useing bbox mode
            sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask  = self.rsp_coarse.forward(rays[:,:6], bboxes)
            sampled_rays_coarse_t = sampled_rays_coarse_t[ray_mask]
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
            rays_t = rays[ray_mask].detach()

            if self.use_deform:
                Rts = Rts[ray_mask].detach()
                global_ts = global_ts[ray_mask].detach()
                skeletons = skeletons[ray_mask].detach()

        if rays_t.size(0) > 1:
            sampled_rays_coarse_t = sampled_rays_coarse_t.detach() # (N, L1, 3)
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach() # (N, L1, 3)

            if self.use_deform:
               
                #Linear Blend Skinning
                lbs_rays_coarse_xyz , w_mask = self.inv_LBS_warper(Rts, global_ts, sampled_rays_coarse_xyz, vertices)

                #filter rays sample points
                msampled_rays_coarse_xyz = sampled_rays_coarse_xyz[w_mask]
                
                ray_dirs = rays_t.unsqueeze(1).repeat(1, sampled_rays_coarse_xyz.shape[1], 1)
                ray_dirs = ray_dirs[w_mask]

                #extract features
                features, angle = self.feature_extractor(feature_maps, msampled_rays_coarse_xyz, scene)

                # blend features
                blend_features = self.attenFeatNet(features, ray_dirs, angle)
                blend_features = torch.sum(blend_features, dim=1) #/ 6
                
                #non-rigid transformation
                offset = self.non_rigid_warper(skeletons, msampled_rays_coarse_xyz, blend_features)

                msampled_rays_coarse_xyz = lbs_rays_coarse_xyz[w_mask] + offset

            #print('COARSE SAMPLING: ', sampled_rays_coarse_xyz.shape)
            rgbs = torch.zeros(sampled_rays_coarse_t.shape[0], sampled_rays_coarse_t.shape[1],3, device = sampled_rays_coarse_t.device)
            density = torch.zeros(sampled_rays_coarse_t.shape[0], sampled_rays_coarse_t.shape[1],1, device = sampled_rays_coarse_t.device)
            mrgbs, mdensity  = self.spacenet(msampled_rays_coarse_xyz, ray_dirs , blend_features, self.maxs, self.mins)
            rgbs[w_mask] = mrgbs
            density[w_mask] = mdensity

            color_0, depth_0, acc_map_0, weights_0 = self.volume_render(sampled_rays_coarse_t, rgbs, density, torch.norm(rays_t[:,:3],dim=-1,keepdim = True))
  
            #torch.cuda.synchronize()
            #print('render coarse:',time.time()-beg)
            if not only_coarse:
                z_samples = sample_pdf(sampled_rays_coarse_t.squeeze(), weights_0.squeeze()[...,1:-1], N_samples = self.fine_ray_sample)
                z_samples = z_samples.detach()   # (N,L)

                z_vals_fine, _ = torch.sort(torch.cat([sampled_rays_coarse_t.squeeze(), z_samples], -1), -1) #(N, L1+L2)
                samples_fine_xyz = z_vals_fine.unsqueeze(-1)*rays_t[:,:3].unsqueeze(1) + rays_t[:,3:6].unsqueeze(1)  # (N,L1+L2,3)

                if self.use_deform:
                    
                    #Linear Blend Skinning 
                    lbs_fine_xyz , w_mask_fine = self.inv_LBS_warper(Rts, global_ts, samples_fine_xyz, vertices)

                    #filter rays sample points
                    msamples_fine_xyz = samples_fine_xyz[w_mask_fine]
                    
                    ray_dirs_fine = rays_t.unsqueeze(1).repeat(1, samples_fine_xyz.shape[1], 1)
                    ray_dirs_fine = ray_dirs_fine[w_mask_fine]

                    #extract features
                    features_fine, angle_fine = self.feature_extractor(feature_maps, msamples_fine_xyz, scene)

                    # blend features
                    # weights_fine = self.blendnet(angle_fine, ray_dirs_fine, features_fine.reshape(-1, features_fine.size(1)*features_fine.size(2)))
                    # blend_features_fine = torch.sum(weights_fine.unsqueeze(-1) * features_fine, dim = 1).
      
                    blend_features_fine = self.attenFeatNet(features_fine, ray_dirs_fine, angle_fine)
                    blend_features_fine = torch.sum(blend_features_fine, dim=1) #/ 6
                    
                    #non-rigid transformation
                    offset_fine  = self.non_rigid_warper(skeletons, msamples_fine_xyz, blend_features_fine)

                    msamples_fine_xyz = lbs_fine_xyz[w_mask_fine] + offset_fine

                rgbs_fine = torch.zeros(z_vals_fine.shape[0], z_vals_fine.shape[1],3, device = z_vals_fine.device)
                density_fine = torch.zeros(z_vals_fine.shape[0], z_vals_fine.shape[1],1, device = z_vals_fine.device)
                mrgbs_fine, mdensity_fine = self.spacenet_fine(msamples_fine_xyz, ray_dirs_fine, blend_features_fine, self.maxs, self.mins)
                rgbs_fine[w_mask_fine] = mrgbs_fine
                density_fine[w_mask_fine] = mdensity_fine

                color, depth, acc_map, weights  = self.volume_render(z_vals_fine.unsqueeze(-1), rgbs_fine, density_fine, torch.norm(rays_t[:,:3],dim=-1, keepdim = True))


                if  self.sample_method == 'BBOX':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                elif self.sample_method == 'DEPTH':
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0


            else:
                if  self.sample_method == 'BBOX':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                elif self.sample_method == 'DEPTH':
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0
                color, depth, acc_map= color_0, depth_0, acc_map_0


            if self.sample_method == 'BBOX':
                color_final = torch.zeros(rays.size(0),3,device = rays.device)
                color_final[ray_mask] = color
                depth_final = torch.zeros(rays.size(0),1,device = rays.device)
                depth_final[ray_mask] = depth
                acc_map_final = torch.zeros(rays.size(0),1,device = rays.device)
                acc_map_final[ray_mask] = acc_map
            elif self.sample_method == 'DEPTH':
                color_final, depth_final, acc_map_final  = color, depth, acc_map
            else:
                color_final , depth_final, acc_map_final = color, depth, acc_map

        else:
            color_final_0 = torch.zeros(rays.size(0),3,device = rays.device).requires_grad_()
            depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
            acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
            weights_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
            color_final, depth_final, acc_map_final, weights_final = color_final_0, depth_final_0, acc_map_final_0, weights_final_0

        return (color_final, depth_final, acc_map_final, ret_final) , (color_final_0, depth_final_0, acc_map_final_0, ret_final_0,), ray_mask

    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins


    def feature_extractor(self, feature_maps, xyz, scene):

        xyz = xyz.transpose(1, 0)
        Ks = scene['Ks'].clone()
        Ts = scene['Ts'].clone()
             
        features = []
        local_dirs = []
        for i in range(Ts.shape[0]):
            RT = Ts[i].inverse()
             
            local_xyz = torch.matmul(RT[:3,:3], xyz) + RT[:3,3].reshape((3,1))

            local_dir = local_xyz / torch.norm(local_xyz, dim=0, keepdim=True)
            local_dirs.append(local_dir)

            local_xyz = torch.matmul(Ks[i], local_xyz)
            local_xyz = (local_xyz / (local_xyz[2, :].unsqueeze(0) + np.finfo(float).eps)).transpose(1,0)
       
            feature = self.bilinear_sample_function(local_xyz, feature_maps[i],  'border')[0]
            features.append(feature.unsqueeze(0))   

        features = torch.cat(features, dim=0)
        local_dirs = torch.cat(local_dirs, dim=0)

        return features.permute(2,0,1), local_dirs.permute(1,0)

    def bilinear_sample_function(self, xyz, feature_map, padding_mode):
        
        c, h, w = feature_map.shape[:3]

        xx = xyz[:, 0].unsqueeze(-1)
        yy = xyz[:, 1].unsqueeze(-1)
        
        xx_norm = xx / (w - 1) * 2 - 1
        yy_norm = yy / (h - 1) * 2 - 1
        
        pixel_coords = torch.stack([xx_norm, yy_norm], dim=2).reshape((1, -1,  1,  2))  # [B, H*W, 2]
        feature = F.grid_sample(feature_map.unsqueeze(0), pixel_coords, padding_mode=padding_mode)[:,:,:,0]
        return feature

    def inv_LBS_warper(self, Rts, global_ts, samples_xyz, vertices):

        bone_weights, w_mask = self.smplBody.query_weights(samples_xyz, vertices)
        if self.use_identity:
            bone_weights = torch.cat([bone_weights, 1. - torch.sum(bone_weights,dim=-1, keepdim=True)],dim=-1)  
            Rts = torch.cat([Rts, self.Rt_identity_flatten.unsqueeze(1).repeat(Rts.shape[0],1,1)], dim=1)
  
        Rt = Rts[0].reshape(-1, 4, 4)
        Rt = Rt.inverse()
        global_t = global_ts[0]

        #Inversed Linear Blend Skinning
        wcanonical_Rts = torch.matmul(bone_weights, self.canonical_Rts).reshape(bone_weights.shape[0], bone_weights.shape[1], 4, 4) # (N, L1+L2, 4, 4)
        wRts = torch.matmul(bone_weights, Rt.reshape(-1, 16)).reshape(bone_weights.shape[0], bone_weights.shape[1], 4, 4) # (N, L1+L2, 4, 4)
        wRts = torch.matmul(wcanonical_Rts, wRts)

        lbs_xyz = torch.sum((samples_xyz - global_t)[:,:,None,:] * wRts[:,:,:3,:3], dim=-1) + wRts[:,:,:3,3]
        
        return lbs_xyz, w_mask

    def non_rigid_warper(self, skeletons, samples_xyz, features):

        rel_offset = samples_xyz[:, None, :] - skeletons[:1, :, :] #(N, 24, 3)
        rel_dist = torch.norm(rel_offset, dim=-1)  #(N, 24)
        rel_dir = rel_offset / (rel_dist.unsqueeze(-1) + 1e-5) #(N,24,3)
        rel_dir = rel_dir.reshape(-1, 24*3)
        offset = self.deformnet(rel_dist, rel_dir, features) #(N,L1+L2,3)

        return offset


    def non_rigid_warper_latent(self, samples_xyz, vertices, indices):
        N, L = samples_xyz.shape[:2]
        smplbody_features = torch.index_select(self.smplbody_features, 0, indices.flatten()) # (N*L, 128)
        smplbody_features = smplbody_features.reshape((N, L, -1))# (N, L, 128)
        
        smplbody_vertices = torch.index_select(vertices, 0, indices.flatten())
        smplbody_vertices = smplbody_vertices.reshape((N, L, 3)) # (N, L, 3)
     
        rel_offset = smplbody_vertices - samples_xyz #(N,L1+L2,3)
        rel_dist = torch.norm(rel_offset, dim=-1, keepdim = True)  #(N,L1+L2)
        rel_dir = rel_offset / (rel_dist + 1e-5) #(N,L1+L2,3)

        offset = self.deformnet(rel_dist, rel_dir, smplbody_features) #(N,L,3)

        return offset