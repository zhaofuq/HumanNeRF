import torch
import cv2
import numpy as np
import os
import json
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torchvision.transforms as TF

import torch.distributions as tdist

from .ibr_dynamic import IBRDynamicDataset
from utils import ray_sampling


class Dataset_NHR(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg , bunch = 1024 ):
        super(Dataset_NHR, self).__init__()

        # configuration
        self.bunch = bunch
        self.use_mask = cfg.DATASETS.USE_MASK
        self.start_frame =  cfg.DATASETS.START_FRAME
        self.num_frame =  cfg.DATASETS.NUM_FRAME
        self.use_depth_proxy = cfg.DATASETS.USE_DEPTH_PROXY
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_bg = cfg.DATASETS.USE_BACKGROUND
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.mask_sampling = cfg.DATASETS.MASK_SAMPLING
        self.split_rays = cfg.DATASETS.SPLIT_RAYS
        self.skip = cfg.DATASETS.SKIP

        #NHR datasets
        self.NHR_dataset = IBRDynamicDataset(data_folder_path, self.start_frame, self.num_frame, self.use_mask, self.use_bg, 
                                            transforms, [1.0, 6.5, 0.8], skip_step = self.skip, random_noisy = 0, holes='None')

        # select cams
        self.num_cam = self.NHR_dataset.cam_num
        self.selected_cams = self.NHR_dataset.selected_cams

        # NeRF data
        self.rays = []
        self.rgbs = []
        self.near_fars = []
        self.frame_ids = []
        self.cam_ids = []
        self.vs = []

        # Aticulated NeRF data
        self.skeletons = []

        self.Rts = []
        self.global_ts = []
        self.bgs = []
        self.depths = []
        
        self.folder =f'rays_temp_{self.num_frame}' if self.start_frame==0 else f'rays_temp_s{self.start_frame}_n{self.num_frame}'
        
        if not os.path.exists(os.path.join(data_folder_path,self.folder)):
            os.mkdir(os.path.join(data_folder_path,self.folder))

        #BUILD ALL RAYS
        if not os.path.exists(os.path.join(os.path.join(data_folder_path,self.folder),'rays_0.pt')):
            for i in range(len(self.NHR_dataset)):

                # img [rgb, mask, ROI, bg] 8 channels
                img, vs, frame_id, T, K, near_far,_,_ = self.NHR_dataset.__getitem__(i)

                self.vs.append(vs)

                img_rgb = img[0:3,:,:] #0-3 RGB
                if self.use_mask:
                    mask = img[4,:,:] * img[3,:,:] # 3-mask, 4-ROI
                    #img_rgb[:, mask<0.5] = 1.0

                rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = img[4,:,:].unsqueeze(0), mask_threshold = 0.5, images = img_rgb.unsqueeze(0))
                
                self.rays.append(rays)
                self.rgbs.append(rgbs)
                #self.cam_ids.append(torch.ones(1,1) * i)
                self.frame_ids.append(torch.ones(rays.size(0),1) * frame_id)        #(N,1)
                self.near_fars.append(near_far.repeat(rays.size(0),1))   # (N,2)
                
                print(frame_id,'| generate %d rays.'%rays.size(0))

            self.vs = torch.cat(self.vs, dim=0)
            self.rays = torch.cat(self.rays, dim=0)
            self.rgbs = torch.cat(self.rgbs, dim=0)
            self.near_fars = torch.cat(self.near_fars, dim=0)   #(N,2)
            # self.cam_ids = torch.cat(self.cam_ids, dim=0)
            self.frame_ids = torch.cat(self.frame_ids, dim=0)  
            
            torch.save(self.rays, os.path.join(os.path.join(data_folder_path,self.folder),'rays_0.pt'))
            torch.save(self.rgbs, os.path.join(os.path.join(data_folder_path,self.folder),'rgb_0.pt'))
            torch.save(self.near_fars, os.path.join(os.path.join(data_folder_path,self.folder),'near_fars_0.pt'))
            torch.save(self.frame_ids, os.path.join(os.path.join(data_folder_path,self.folder),'frameid_0.pt'))
            # torch.save(self.cam_ids, os.path.join(os.path.join(data_folder_path,self.folder),'camid_0.pt'))
          
        else:
            self.rays = torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'rays_0.pt'))
            self.rgbs = torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'rgb_0.pt'))
            self.near_fars =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'near_fars_0.pt'))
            self.frame_ids =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'frameid_0.pt'))
            # self.cam_ids =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'camid_0.pt'))

            img, self.vs, _, T, K, _ ,_ ,_= self.NHR_dataset.__getitem__(0)
            print('load %d rays.'%self.rays.size(0))

        #-------------------------------------------------------------------ADDITIONAL DATA-------------------------------------------------------------------------#

        if self.use_bg:
            if not os.path.exists(os.path.join(os.path.join(data_folder_path,self.folder),'bg_0.pt')):
                for i in range(len(self.NHR_dataset)):

                    # img [rgb, mask, ROI, bg] 8 channels
                    img, _, _, _, _, _,_,_ = self.NHR_dataset.__getitem__(i)
                    bg = img[5:,:,:] # 5-8 BACKGROUND
                    bg = bg[:,img[4,:,:] > 0.5].reshape(3,-1).permute(1,0)
                    self.bgs.append(bg)
                    print(i,'| generate backgrounds {}.'.format(bg.shape))
                self.bgs = torch.cat(self.bgs, dim=0)
                torch.save(self.bgs, os.path.join(os.path.join(data_folder_path,self.folder),'bg_0.pt'))
            else:
                self.bgs =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'bg_0.pt'))

        if self.use_skeleton:
            assert os.path.exists(os.path.join(data_folder_path, 'skeletons')), 'skeleton parameters not exists!'

            if not os.path.exists(os.path.join(os.path.join(data_folder_path,self.folder),'Rt_0.pt')):

                for i in range(self.start_frame, self.start_frame + self.num_frame):
                    Rt = torch.tensor(np.load(os.path.join(data_folder_path,'skeletons/Rt%d.npy' % i)).astype(np.float32)).unsqueeze(0)
                    global_t = torch.tensor(np.load(os.path.join(data_folder_path,'skeletons/global_t%d.npy' % i)).astype(np.float32)).unsqueeze(0)
                    skeleton = torch.tensor(np.load(os.path.join(data_folder_path,'skeletons/skeletonpose%d.npy' % i)).astype(np.float32)).unsqueeze(0)

                    self.Rts.append(Rt)
                    self.global_ts.append(global_t)
                    self.skeletons.append(skeleton)
                    
                self.Rts = torch.cat(self.Rts, dim=0)
                self.global_ts = torch.cat(self.global_ts, dim=0)
                self.skeletons = torch.cat(self.skeletons, dim=0)

                torch.save(self.Rts, os.path.join(os.path.join(data_folder_path,self.folder),'Rt_0.pt'))
                torch.save(self.global_ts, os.path.join(os.path.join(data_folder_path,self.folder),'global_t_0.pt'))
                torch.save(self.skeletons, os.path.join(os.path.join(data_folder_path,self.folder),'skeleton_t_0.pt'))
            else:

                self.Rts =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'Rt_0.pt'))
                self.global_ts =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'global_t_0.pt'))
                self.skeletons =  torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'skeleton_t_0.pt'))

        if self.use_depth_proxy:
            ts = torchvision.transforms.ToPILImage(mode='F')

            if os.path.exists(os.path.join(os.path.join(data_folder_path,self.folder),'train_depth.pt')):
                self.depths = torch.load(os.path.join(os.path.join(data_folder_path,self.folder),'train_depth.pt'))
            else:
                camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
                Ts = torch.Tensor( campose_to_extrinsic(camposes) )
                Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

                for i in range(self.start_frame,  self.start_frame + self.num_frame, self.skip):
                    for cid in range(self.num_cam):
                        cam_id = self.selected_cams[cid]
                        depth_raw = ts(torch.tensor(torch.load(os.path.join(data_folder_path,f'depth/{i}/depth_{cam_id:04d}.pt'))).float())
                        #depth_raw = ts(torch.load(os.path.join(data_folder_path,f'depth/depth_{(i * 10 + cam_id):04d}.pt')))
                        
                        depth, _, _, _, ROI = transforms(depth_raw, Ks[cam_id], Ts[cam_id])
                        self.depths.append(depth[ROI>0.5].unsqueeze(-1))
                        print(i,'| generate depth map {}.'.format(depth[ROI>0.5].shape))

                self.depths = torch.cat(self.depths, dim=0)
                torch.save(self.depths,os.path.join(os.path.join(data_folder_path, self.folder),'train_depth.pt'))

        #sparse background sampling
        if self.mask_sampling and self.use_depth_proxy: 
            self.depths = self.depths[:self.rays.shape[0],:]
            rgb_mask = self.depths.squeeze(-1) > 0.5
            bg_mask = self.depths.squeeze(-1) < 0.
            bg_mask[::5] = True
            rgb_mask = rgb_mask + bg_mask

            self.rays = self.rays[rgb_mask] 
            self.rgbs = self.rgbs[rgb_mask] 
            self.near_fars = self.near_fars[rgb_mask]
            self.frame_ids = self.frame_ids[rgb_mask]
            self.depths = self.depths[rgb_mask]

        #-------------------------------------------------------------------ADDITIONAL DATA-------------------------------------------------------------------------#    

        #bbox
        max_xyz = torch.max(self.vs, dim=0)[0]
        min_xyz = torch.min(self.vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * 0.3

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))

        if self.split_rays:
            self.bunch_split_rays()

    
    def bunch_split_rays(self):

        self.rays = self.rays.reshape(self.num_frame, -1, 6)
        self.rgbs = self.rgbs.reshape(self.num_frame, -1, 3)
        self.near_fars = self.near_fars.reshape(self.num_frame, -1, 2)

        if self.use_depth_proxy:
            self.depths = self.depths.reshape(self.num_frame, -1, 1)

        if self.use_bg:
            self.bgs = self.bgs.reshape(self.num_frame, -1, 3)


    def __len__(self):
        if self.split_rays:
            return self.num_frame * self.rays.size(1) // self.bunch
        else:
            return self.rays.size(0)
    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):
        sample = {}

        if self.split_rays:
            frame_id = np.random.choice(self.num_frame, size = 1, replace=True)[0]
            idx = np.random.choice(self.rays.size(1), size=self.bunch, replace=True)
            
            sample['rays'] = torch.cat([self.rays[frame_id, idx], frame_id * torch.ones((self.bunch, 1))], dim=-1)
            
            sample['rgbs'] = self.rgbs[frame_id, idx]
            sample['bboxes'] =  self.bbox[0]
            sample['near_fars'] = self.near_fars[frame_id, idx]
            sample['frame_ids'] =   frame_id * torch.ones((self.bunch, 1))

            if self.use_depth_proxy:
                sample['depth'] = self.depths[frame_id, idx]

            if self.use_bg:
                sample['bgs'] = self.bgs[frame_id, idx]

            if self.use_skeleton:
                sample['Rts'] = self.Rts[frame_id,:].unsqueeze(0).repeat(self.bunch,1,1) 
                sample['global_ts'] = self.global_ts[frame_id,:].unsqueeze(0).repeat(self.bunch, 1) 

                sample['skeletons'] = self.skeletons[frame_id,:].unsqueeze(0).repeat(self.bunch,1,1) 

        else:
            frame_id = self.frame_ids[index,:].long()
            #cam_id = index * len(self.NHR_dataset) // self.rays.size(0) % self.num_cam

            sample['rays'] = torch.cat([self.rays[index,:], torch.tensor([frame_id]).float()])
            sample['rgbs'] = self.rgbs[index,:]
            sample['bboxes'] =  self.bbox[0]
            sample['near_fars'] = self.near_fars[index,:]
            sample['frame_ids'] = torch.tensor([frame_id])       
            #sample['cam_ids'] = torch.tensor([cam_id])    

            if self.use_depth_proxy:
                sample['depth'] = self.depths[index,:]

            if self.use_bg:
                sample['bgs'] = self.bgs[index,:]

            if self.use_skeleton:
                sample['Rts'] = self.Rts[frame_id,:]
                sample['global_ts'] = self.global_ts[frame_id,:]

                if frame_id <2:
                    sample['skeletons'] = torch.cat([self.skeletons[frame_id,:], self.skeletons[frame_id,:],self.skeletons[frame_id,:]], dim=0).reshape(-1,3)
                else:
                    sample['skeletons'] = torch.cat([self.skeletons[frame_id-2,:], self.skeletons[frame_id-1,:],self.skeletons[frame_id,:]], dim=0).reshape(-1,3)
         
        return sample

class Dataset_NeRF(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg , bunch = 3600):
        super(Dataset_NeRF, self).__init__()

        # configuration
        self.bunch = bunch
        self.data_folder_path = data_folder_path
        self.use_mask = cfg.DATASETS.USE_MASK
        self.start_frame =  cfg.DATASETS.START_FRAME
        self.num_frame =  cfg.DATASETS.NUM_FRAME
        self.use_depth_proxy = cfg.DATASETS.USE_DEPTH_PROXY
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_bg = cfg.DATASETS.USE_BACKGROUND
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.mask_sampling = cfg.DATASETS.MASK_SAMPLING
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0])

        #NHR datasets
        self.NHR_dataset = IBRDynamicDataset(data_folder_path, self.start_frame, self.num_frame, False, self.use_bg, 
                                            transforms, [1.0, 6.5, 0.8], skip_step = self.skip, random_noisy = 0, holes='None')
        
        # select cams
        self.num_cam = self.NHR_dataset.cam_num
        self.selected_cams = self.NHR_dataset.selected_cams

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

        #shuffle rays indices
        self.shuffle_all_rays()


    def __len__(self):

        return len(self.all_rays)

    def shuffle_all_rays(self):

        if not os.path.exists(os.path.join(self.data_folder_path,f'shufle_rays_{len(self.NHR_dataset)}.pt')):
            all_rays = []
            print('shuffle rays...')
            for i in range(len(self.NHR_dataset)):
                all_rays.append(torch.randperm(self.size[0]*self.size[1]))
            all_rays = torch.cat(all_rays, dim=0)
            self.all_rays = all_rays.split(self.bunch)
            torch.save(self.all_rays, os.path.join(self.data_folder_path,f'shufle_rays_{len(self.NHR_dataset)}.pt'))
            print('rays shuffled...')
        else:
            print('load shuffled rays...')
            self.all_rays = torch.load(os.path.join(self.data_folder_path,f'shufle_rays_{len(self.NHR_dataset)}.pt'))

    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    def build_batch_rays(self, frame_id, cam_id):

        img = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/img_{cam_id:04d}.jpg'))
        #build rays
        T = self.Ts[cam_id].clone()
        K = self.Ks[cam_id].clone()
        K[:2,:] = K[:2,:] * self.size[1] / img.size[0]   

        img = TF.functional.to_tensor(img)
        img = TF.functional.resize(img, self.size)

        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = torch.ones(img[0,:,:].size()).unsqueeze(0), mask_threshold = 0.5, images = img.unsqueeze(0))

        if self.use_mask:
            mask = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/mask_{cam_id:04d}.jpg'))
            mask = TF.functional.to_tensor(mask)
            mask = TF.functional.resize(mask, self.size)
            mask = mask[:1,:,:].reshape(-1,1)
            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0
        else:
            mask = None

        if self.use_depth_proxy:
            depth = torch.load(os.path.join(self.data_folder_path,f'depth/{frame_id}/depth_{cam_id:04d}.pt')).float()
            depth = TF.functional.resize(depth.unsqueeze(0), self.size)
            depth = depth.reshape(-1,1)
        else:
            depth = None

        if self.use_bg:
            bg = Image.open(os.path.join(self.data_folder_path,f'img/background/img_{cam_id:04d}.jpg'))
            bg = TF.functional.to_tensor(bg)
            bg = TF.functional.resize(bg, self.size)
            bg = bg.reshape(-1,3)
        else:
            bg = None

        return rays, rgbs, mask, depth ,bg

    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):
        sample = {}
        nhr_index = index * self.bunch // (self.size[0] * self.size[1])
        frame_id = ((nhr_index // self.num_cam) * self.skip) % self.num_frame 
        cam_id = self.selected_cams[nhr_index % self.num_cam]

        #build rays
        rays, rgbs, mask, depth ,bg = self.build_batch_rays(frame_id, cam_id)
        ray_idx = self.all_rays[index]

        frame_ids = torch.ones((self.bunch,1))* frame_id
        cam_ids = torch.ones((self.bunch,1))* cam_id

        sample['rays'] = rays[ray_idx,:]
        sample['rgbs'] = rgbs[ray_idx,:]
        sample['near_fars'] = torch.tensor([self.NHR_dataset.near[cam_id],self.NHR_dataset.far[cam_id]]).unsqueeze(0).repeat(self.bunch, 1)

        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'pointclouds/frame{frame_id + 1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(self.bunch, 1, 1)

        sample['frame_ids'] = frame_ids    
        sample['cam_ids'] = cam_ids   

        if self.use_mask:
            sample['mask'] = mask[ray_idx,:]

        if self.use_depth_proxy:
            sample['depth'] = depth[ray_idx,:]
     
        if self.use_bg:
            sample['bgs'] = bg[ray_idx,:]


        if self.use_skeleton:
            sample['Rts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/Rt{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1,1) 
            sample['global_ts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/global_t{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1) 
            sample['skeletons'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/skeletonpose{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1,1) 
        return sample

class Dataset_View(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg):
        super(Dataset_View, self).__init__()

        self.data_folder_path = data_folder_path 
        self.transforms = transforms
        
        self.use_mask = cfg.DATASETS.USE_MASK
        self.start_frame =  cfg.DATASETS.START_FRAME
        self.num_frame =  cfg.DATASETS.NUM_FRAME
        self.use_depth_proxy = cfg.DATASETS.USE_DEPTH_PROXY
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_bg = cfg.DATASETS.USE_BACKGROUND
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0])

        self.NHR_dataset = IBRDynamicDataset(data_folder_path, self.start_frame , self.num_frame, False, self.use_bg, 
                                            transforms, [1.0, 6.5, 0.8], skip_step = self.skip, random_noisy = 0, holes='None')
        
        img, self.vs, _, T, K, _,_ ,_= self.NHR_dataset.__getitem__(0)


        max_xyz = torch.max(self.vs, dim=0)[0]
        min_xyz = torch.min(self.vs, dim=0)[0]

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]

        tmp = (max_xyz - min_xyz) * 0.3

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
        

    def __len__(self):
        return 1

    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    def __getitem__(self, index):

        index = np.random.randint(0,len(self.NHR_dataset))
        #index = index % len(self.NHR_dataset) * self.skip
        img, self.vs, frame_id, T, K, near_far,_ , cam_id = self.NHR_dataset.__getitem__(index)

        img_rgb = img[0:3,:,:]
        if self.use_mask:
            mask = img[4,:,:] *img[3,:,:] 
            img_rgb[:, mask<0.5] = 1.0
            
        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), images = img_rgb.unsqueeze(0))

        frame_ids = torch.ones((rays.size(0),1))* frame_id
        cam_ids = torch.ones((rays.size(0),1))* cam_id
        rays = torch.cat([rays,frame_ids],dim=1)

        sample = {}

        sample['rays'] = rays
        sample['rgbs'] = rgbs
        sample['color'] = img_rgb
        sample['mask'] = img[3,:,:].unsqueeze(0)
        sample['ROI'] = img[4,:,:].unsqueeze(0)
        sample['near_fars'] = near_far.repeat(rays.size(0), 1)

        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'pointclouds/frame{frame_id+1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(rays.size(0), 1, 1)

        sample['frame_ids'] = frame_ids
        sample['cam_ids'] = cam_ids
        
        if self.use_mask:
            mask = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id }/mask_{cam_id:04d}.jpg'))
            mask = TF.functional.to_tensor(mask)
            mask = TF.functional.resize(mask, self.size)
            sample['mask'] = mask[:1,:,:]

        if self.use_depth_proxy:
            depth = torch.tensor(torch.load(os.path.join(self.data_folder_path,f'depth/{frame_id+self.start_frame}/depth_{cam_id:04d}.pt'))).float()
            depth = TF.functional.resize(depth.unsqueeze(0), self.size)
            sample['depth'] = depth.reshape(-1,1)

        if self.use_bg:
            sample['bgs'] = img[5:,:,:].reshape(3,-1).permute(1,0)

        if self.use_skeleton:
            Rts = torch.tensor(np.load(os.path.join(self.data_folder_path, 'skeletons/Rt%d.npy' % (frame_id) )).astype(np.float32))
            sample['Rts'] = Rts.reshape(1, -1, 16).repeat(rays.size(0),1,1)
            global_ts = torch.tensor(np.load(os.path.join(self.data_folder_path, 'skeletons/global_t%d.npy' % (frame_id) )).astype(np.float32))
            sample['global_ts'] = global_ts.reshape(1, 3).repeat(rays.size(0),1)
            skeletons = torch.tensor(np.load(os.path.join(self.data_folder_path, 'skeletons/skeletonpose%d.npy' % (frame_id) )).astype(np.float32))
            sample['skeletons'] = skeletons.reshape(1, 24, 3).repeat(rays.size(0),1,1)

        return sample

class Dataset_MultiNeRF(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg , bunch = 3600):
        super(Dataset_MultiNeRF, self).__init__()

        # configuration
        self.bunch = bunch
        self.data_folder_path = data_folder_path
        self.use_mask = cfg.DATASETS.USE_MASK
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0])
        self.finetune = cfg.DATASETS.FINETUNE

        #prepare scene
        self.prepare_scene()

    def __len__(self):

        return len(self.scenes)
    
    def prepare_scene(self):
        if self.finetune:
            self.scenes = json.load(open(os.path.join(self.data_folder_path, 'finetune.json')))
        else:
            self.scenes = json.load(open(os.path.join(self.data_folder_path, 'train.json')))
    
    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    def build_cams(self, scene):
        Ts = campose_to_extrinsic(np.loadtxt(os.path.join(self.data_folder_path,f'{scene}/CamPose.inf')))
        Ks = read_intrinsics(os.path.join(self.data_folder_path,f'{scene}/Intrinsic.inf'))
        Ts = torch.from_numpy(Ts).float()
        Ks = torch.from_numpy(Ks).float()

        Ts_120 = campose_to_extrinsic(np.loadtxt(os.path.join(self.data_folder_path,f'{scene}/CamPose_120.inf')))
        Ks_120 = read_intrinsics(os.path.join(self.data_folder_path,f'{scene}/Intrinsic_120.inf'))
        Ts_120 = torch.from_numpy(Ts_120).float()
        Ks_120 = torch.from_numpy(Ks_120).float()

        Ks = torch.cat([Ks, Ks_120], dim=0)
        Ts = torch.cat([Ts, Ts_120], dim=0)

        return Ks, Ts
    
    def sample_imgs(self, scene, frame_id, cam_id):

        imgs = []
        #nearest = torch.argmin(torch.norm(FTs[cam_id,:3,3] - Ts[:,:3,3], dim=-1))
        #cams = [(Ts.size(0) - 1  + nearest) % Ts.size(0), nearest, (Ts.size(0) + 1 + nearest) % Ts.size(0)]

        # if cam_id < 6:
        #     cams = [cam_id, (cam_id+1) % 6]
        # else:
        #     cams = [(cam_id -6) // 20, ((cam_id-6) // 20 + 1) % 6]
        cams = [0, 1, 2, 3, 4, 5]

        for i in cams:
            img = Image.open(os.path.join(self.data_folder_path ,f'{scene}/img/{frame_id}/img_{i:04d}.jpg'))
            img = TF.functional.to_tensor(img)
            img = TF.functional.resize(img, (256,256))

            mask = Image.open(os.path.join(self.data_folder_path,f'{scene}/img/{frame_id}/mask_{i:04d}.jpg'))
            mask = TF.functional.to_tensor(mask)
            mask = TF.functional.resize(mask[:1,:,:], (256,256))

            img = torch.cat([img, mask], dim=0)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)

        image = Image.open(os.path.join(self.data_folder_path ,f'{scene}/img/{frame_id}/img_{cam_id:04d}.jpg'))
        image = TF.functional.to_tensor(image)
        
        mask = Image.open(os.path.join(self.data_folder_path,f'{scene}/img/{frame_id}/mask_{cam_id:04d}.jpg'))
        mask = TF.functional.to_tensor(mask)
        
        return imgs, image, mask, cams

    def sample_rays(self, img, mask, K, T):

        K[:2,:] = K[:2,:] *  self.size[1] / img.shape[1]

        img = TF.functional.resize(img, self.size)
        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = torch.ones(img[0,:,:].size()).unsqueeze(0), mask_threshold = 0.5, images = img.unsqueeze(0))

        mask = TF.functional.resize(mask, self.size)
        mask = mask[:1,:,:].reshape(-1,1)
        
        return rays, rgbs, mask

    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):
        sample = {}
        # nhr_index = index * self.bunch // (self.size[0] * self.size[1])
        # frame_id = ((nhr_index // self.num_cam) * self.skip) % self.num_frame 
        # cam_id = self.selected_cams[nhr_index % self.num_cam]
        #  ray_idx = self.all_rays[index]
        scene = self.scenes[index]
        scene_name = scene['scene']
        frame_id = scene['frame_id']
        cam_id = scene['cam_id']

        Ks, Ts = self.build_cams(scene_name)
        img_list, image, mask, select_cams = self.sample_imgs(scene_name, frame_id, cam_id)
        rays, rgbs, mask = self.sample_rays(image, mask, Ks[cam_id].clone(), Ts[cam_id].clone())

        ray_idx = np.random.choice(self.size[0]*self.size[1], self.bunch, replace=False)

        sample['rays'] = rays[ray_idx,:]
        sample['rgbs'] = rgbs[ray_idx,:]
        sample['mask'] = mask[ray_idx,:]
        sample['near_fars'] = torch.tensor([0.5, 5.0]).unsqueeze(0).repeat(self.bunch, 1)

        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'{scene_name}/pointclouds/frame{frame_id + 1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(self.bunch, 1, 1)

        sample['frame_ids'] = torch.ones((self.bunch,1))* frame_id    
        sample['cam_ids'] = torch.ones((self.bunch,1))* cam_id   

        sample['scene'] = scene_name

        sample['Ks'] = Ks[select_cams].clone()
        sample['Ks'][:, :2, :] = sample['Ks'][:, :2, :] * 256 / image.size(1)
        sample['Ts'] = Ts[select_cams]
        sample['imgs'] = img_list

        if self.use_skeleton:
            Rts = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/Rt%d.npy' % (frame_id) )).astype(np.float32))
            sample['Rts'] = Rts.reshape(1, -1, 16).repeat(self.bunch,1,1)
            global_ts = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/global_t%d.npy' % (frame_id) )).astype(np.float32))
            sample['global_ts'] = global_ts.reshape(1, 3).repeat(self.bunch,1)
            skeletons = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/skeletonpose%d.npy' % (frame_id) )).astype(np.float32))
            sample['skeletons'] = skeletons.reshape(1, 24, 3).repeat(self.bunch,1,1)

        return sample

class Dataset_MultiView(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg):
        super(Dataset_MultiView, self).__init__()

        # configuration
        self.data_folder_path = data_folder_path
        self.use_mask = cfg.DATASETS.USE_MASK
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0])
        self.finetune = cfg.DATASETS.FINETUNE

        #prepare scene
        self.prepare_scene()

    def __len__(self):
   
        return 1
    
    def prepare_scene(self):
        if self.finetune:
            self.scenes = json.load(open(os.path.join(self.data_folder_path, 'finetune.json')))
        else:
            self.scenes = json.load(open(os.path.join(self.data_folder_path, 'test.json')))

    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    
    def build_cams(self, scene):

        Ts = campose_to_extrinsic(np.loadtxt(os.path.join(self.data_folder_path,f'{scene}/CamPose.inf')))
        Ks = read_intrinsics(os.path.join(self.data_folder_path,f'{scene}/Intrinsic.inf'))
        Ts = torch.from_numpy(Ts).float()
        Ks = torch.from_numpy(Ks).float()

        Ts_120 = campose_to_extrinsic(np.loadtxt(os.path.join(self.data_folder_path,f'{scene}/CamPose_120.inf')))
        Ks_120 = read_intrinsics(os.path.join(self.data_folder_path,f'{scene}/Intrinsic_120.inf'))
        Ts_120 = torch.from_numpy(Ts_120).float()
        Ks_120 = torch.from_numpy(Ks_120).float()

        Ks = torch.cat([Ks, Ks_120], dim=0)
        Ts = torch.cat([Ts, Ts_120], dim=0)

        return Ks, Ts


    def sample_imgs(self, scene, frame_id, cam_id):

        imgs = []
        # nearest = torch.argmin(torch.norm(FTs[cam_id,:3,3] - Ts[:,:3,3], dim=-1))
        # cams = [(Ts.size(0) - 1  + nearest) % Ts.size(0), nearest, (Ts.size(0) + 1 + nearest) % Ts.size(0)]
        # if cam_id < 6:
        #     cams = [cam_id, (cam_id+1) % 6]
        # else:
        #     cams = [(cam_id -6) // 20, ((cam_id-6) // 20 + 1) % 6]
        cams = [0,1,2,3,4,5]

        for i in cams:
            img = Image.open(os.path.join(self.data_folder_path ,f'{scene}/img/{frame_id}/img_{i:04d}.jpg'))
            img = TF.functional.to_tensor(img)
            img = TF.functional.resize(img, (256,256))

            mask = Image.open(os.path.join(self.data_folder_path,f'{scene}/img/{frame_id}/mask_{i:04d}.jpg'))
            mask = TF.functional.to_tensor(mask)
            mask = TF.functional.resize(mask[:1,:,:], (256,256))

            img = torch.cat([img, mask], dim=0)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)

        image = Image.open(os.path.join(self.data_folder_path ,f'{scene}/img/{frame_id}/img_{cam_id:04d}.jpg'))
        image = TF.functional.to_tensor(image)
        
        mask = Image.open(os.path.join(self.data_folder_path,f'{scene}/img/{frame_id}/mask_{cam_id:04d}.jpg'))
        mask = TF.functional.to_tensor(mask)
        
        return imgs, image, mask, cams

    def sample_rays(self, img, mask, K, T):

        K[:2,:] = K[:2,:] *  self.size[1] / img.shape[1]

        img = TF.functional.resize(img, self.size)
        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = torch.ones(img[0,:,:].size()).unsqueeze(0), mask_threshold = 0.5, images = img.unsqueeze(0))

        mask = TF.functional.resize(mask, self.size)
        mask = mask[:1,:,:].reshape(-1,1)
        
        return rays, rgbs, img, mask

    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):

        index = np.random.randint(0,len(self.scenes))

        sample = {}
        scene = self.scenes[index]
        scene_name = scene['scene']
        frame_id = scene['frame_id']
        cam_id = scene['cam_id']
        

        Ks, Ts = self.build_cams(scene_name)
        img_list, image, mask, select_cams = self.sample_imgs(scene_name, frame_id, cam_id)
        rays, rgbs, color,  mask = self.sample_rays(image, mask, Ks[cam_id].clone(), Ts[cam_id].clone())

        sample['rays'] = rays
        sample['rgbs'] = rgbs
        sample['color'] = color
        sample['mask'] = mask

        sample['near_fars'] = torch.tensor([0.5, 5.0]).unsqueeze(0).repeat(rays.size(0), 1)
        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'{scene_name}/pointclouds/frame{frame_id+1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(rays.size(0), 1, 1)

        sample['frame_ids'] = torch.ones((rays.size(0),1))* frame_id
        sample['cam_ids'] = torch.ones((rays.size(0),1))* cam_id
        
        sample['scene'] = scene_name
        sample['Ks'] = Ks[select_cams].clone() #Ks[:6]
        sample['Ks'][:, :2, :] = sample['Ks'][:, :2, :] * 256 / image.size(1)
        sample['Ts'] = Ts[select_cams] #Ts[:6]
        sample['imgs'] = img_list
        
        if self.use_skeleton:
            Rts = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/Rt%d.npy' % (frame_id) )).astype(np.float32))
            sample['Rts'] = Rts.reshape(1, -1, 16).repeat(rays.size(0),1,1)
            global_ts = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/global_t%d.npy' % (frame_id) )).astype(np.float32))
            sample['global_ts'] = global_ts.reshape(1, 3).repeat(rays.size(0),1)
            skeletons = torch.tensor(np.load(os.path.join(self.data_folder_path, f'{scene_name}/skeletons/skeletonpose%d.npy' % (frame_id) )).astype(np.float32))
            sample['skeletons'] = skeletons.reshape(1, 24, 3).repeat(rays.size(0),1,1)

        return sample

class Pointcloud_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_folder_path, sample_num, scale):

        # 1. Check if the pointcloud exists
        # --------------------------------------------------------------------------------------
        if os.path.exists(os.path.join(data_folder_path,'pointclouds/frame1.npy')):
            tmp = np.load(os.path.join(data_folder_path,'pointclouds/frame1.npy'))
            num_channel = tmp.shape[1]

            # 2. Load pointcloud with xyz (and rgb)
            # --------------------------------------------------------------------------------------
            if num_channel == 3:
                print('Warning: The pointcloud only has xyz, treat the color as white') 
                self.points = torch.Tensor(tmp) * scale
                self.colors = torch.ones_like(self.points)
            elif num_channel == 6:
                print('The pointcloud has xyz and rgb')
                self.points = torch.Tensor(tmp[:,0:3]) * scale
                self.colors = torch.Tensor(tmp[:,3:6])
            self.occupys = torch.ones(self.points.shape[0],1)
            print('Load %d vertices from pointcloud' % self.points.shape[0])
            
            # 3. Calculate a near far for randomly sample empty points
            # --------------------------------------------------------------------------------------
            camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
            self.Ts = torch.Tensor(campose_to_extrinsic(camposes))
            self.Ts[:,0:3,3] = self.Ts[:,0:3,3] * scale
            self.cam_num = self.Ts.size(0)
            self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))
            
            inv_Ts = torch.inverse(self.Ts).unsqueeze(1)  #(M,1,4,4)
            vs = self.points.clone().unsqueeze(-1)   #(N,3,1)
            vs = torch.cat([vs,torch.ones(vs.size(0),1,vs.size(2)) ],dim=1) #(N,4,1)

            pts = torch.matmul(inv_Ts,vs) #(M,N,4,1)

            pts_max = torch.max(pts, dim=1)[0].squeeze() #(M,4)
            pts_min = torch.min(pts, dim=1)[0].squeeze() #(M,4)

            pts_max = pts_max[:,2]   #(M)
            pts_min = pts_min[:,2]   #(M)

            self.near = pts_min *0.5
            self.near[self.near<(pts_max*0.1)] = pts_max[self.near<(pts_max*0.1)]*0.1
            
            self.near = torch.Tensor(self.near)
            self.far = pts_max * 1.2
            self.far = torch.Tensor(self.far)

            self.near_fars = torch.stack((self.near,self.far),-1)

            # 4. Calculate bbox
            # --------------------------------------------------------------------------------------
            max_xyz = torch.max(self.points, dim=0)[0]
            min_xyz = torch.min(self.points, dim=0)[0]

            minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
            maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
            bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],
                             [minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])

            self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
            self.empty_points = torch.meshgrid(torch.arange(minx,maxx,(maxx-minx)/sample_num),
                                              torch.arange(miny,maxy,(maxy-miny)/sample_num),
                                              torch.arange(minz,maxz,(maxz-minz)/sample_num))
            self.empty_points = torch.stack(self.empty_points,-1)
            self.empty_points = self.empty_points.reshape(-1,3)
            self.empty_colors = torch.zeros_like(self.empty_points)
            self.empty_occupys = torch.zeros(self.empty_points.shape[0],1)
            # --------------------------------------------------------------------------------------
            #TODO: Add near far empty points
            # 5. Concat points & colors & occupys
            self.points = torch.cat((self.points,self.empty_points),0)
            self.colors = torch.cat((self.colors,self.empty_colors),0)
            self.occupys = torch.cat((self.occupys,self.empty_occupys),0)
        else:
            print('Error: There is no pointcloud in the path %s' % os.path.join(data_folder_path,'pointclouds/frame1.npy'))
        self.camera_num = self.Ts.shape[0]
    
    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index,:], self.colors[index,:], self.occupys[index,:]

class Dataset_NeRF_Batch(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg , bunch = 3600):
        super(Dataset_NeRF_Batch, self).__init__()

        # configuration
        self.bunch = bunch
        self.data_folder_path = data_folder_path
        self.use_mask = cfg.DATASETS.USE_MASK
        self.start_frame =  cfg.DATASETS.START_FRAME
        self.num_frame =  cfg.DATASETS.NUM_FRAME
        self.use_depth_proxy = cfg.DATASETS.USE_DEPTH_PROXY
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_bg = cfg.DATASETS.USE_BACKGROUND
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.mask_sampling = cfg.DATASETS.MASK_SAMPLING
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0])

        #NHR datasets
        self.NHR_dataset = IBRDynamicDataset(data_folder_path, self.start_frame, self.num_frame, False, self.use_bg, 
                                            transforms, [1.0, 6.5, 0.8], skip_step = self.skip, random_noisy = 0, holes='None')
        
        # select cams
        self.num_cam = self.NHR_dataset.cam_num
        self.selected_cams = self.NHR_dataset.selected_cams

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

    def __len__(self):
        return len(self.NHR_dataset)

    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    def build_batch_rays(self, frame_id, cam_id):

        img = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/img_{cam_id:04d}.jpg'))
        #build rays
        T = self.Ts[cam_id].clone()
        K = self.Ks[cam_id].clone()
        K[:2,:] = K[:2,:] * self.size[1] / img.size[0]   

        img = TF.functional.to_tensor(img)
        img = TF.functional.resize(img, self.size)

        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = torch.ones(img[0,:,:].size()).unsqueeze(0), mask_threshold = 0.5, images = img.unsqueeze(0))

        if self.use_mask:
            mask = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/mask_{cam_id:04d}.jpg'))
            mask = TF.functional.to_tensor(mask)
            mask = TF.functional.resize(mask, self.size, Image.NEAREST)
            mask = mask[:1,:,:].reshape(-1,1)
        else:
            mask = None

        if self.use_depth_proxy:
            depth = torch.load(os.path.join(self.data_folder_path,f'depth/{frame_id}/depth_{cam_id:04d}.pt')).float()
            depth = TF.functional.resize(depth.unsqueeze(0), self.size, Image.NEAREST)
            depth = depth.reshape(-1,1)
        else:
            depth = None

        if self.use_bg:
            bg = Image.open(os.path.join(self.data_folder_path,f'img/background/img_{cam_id:04d}.jpg'))
            bg = TF.functional.to_tensor(bg)
            bg = TF.functional.resize(bg, self.size, Image.BICUBIC)
            bg = bg.reshape(-1,3)
        else:
            bg = None

        if self.mask_sampling:
            rgb_mask = mask[:,0] > 0.5
            bg_mask = mask[:,0] < - 0.5
            bg_mask[::5] = True
            rgb_mask = rgb_mask + bg_mask

            rays = rays[rgb_mask]
            rgbs = rgbs[rgb_mask]
            mask = mask[rgb_mask]
            
            depth = depth[rgb_mask] if depth is not None else depth
            bg = bg[rgb_mask] if bg is not None else bg
            

        return rays, rgbs, mask, depth ,bg

    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):
        sample = {}
        nhr_index = index * self.bunch // (self.size[0] * self.size[1])
        frame_id = ((nhr_index // self.num_cam) * self.skip) % self.num_frame 
        cam_id = self.selected_cams[nhr_index % self.num_cam]

        #build rays
        rays, rgbs, mask, depth ,bg = self.build_batch_rays(frame_id, cam_id)
        shuffle_idx = torch.randperm(rays.size(0))
        # frame_ids = torch.ones((self.bunch,1))* frame_id
        # cam_ids = torch.ones((self.bunch,1))* cam_id

        sample['rays'] = rays[shuffle_idx]
        sample['rgbs'] = rgbs[shuffle_idx]
        sample['near_fars'] = torch.tensor([self.NHR_dataset.near[cam_id],self.NHR_dataset.far[cam_id]]).unsqueeze(0).repeat(rays.size(0), 1)

        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'pointclouds/frame{frame_id+1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(rays.size(0), 1, 1)

        # sample['frame_ids'] = frame_ids    
        # sample['cam_ids'] = cam_ids   

        if self.use_mask:
            sample['mask'] = mask[shuffle_idx]

        if self.use_depth_proxy:
            sample['depth'] = depth[shuffle_idx]
     
        if self.use_bg:
            sample['bgs'] = bg[shuffle_idx]


        if self.use_skeleton:
            sample['Rts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/Rt{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(rays.size(0),1,1) 
            sample['global_ts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/global_t{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(rays.size(0),1) 
            sample['skeletons'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/skeletonpose{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(rays.size(0),1,1) 
        return sample

class Dataset_NB(torch.utils.data.Dataset):

    def __init__(self,data_folder_path,  transforms, cfg , bunch = 3600):
        super(Dataset_NB, self).__init__()

        # configuration
        self.bunch = bunch
        self.data_folder_path = data_folder_path
        self.use_mask = cfg.DATASETS.USE_MASK
        self.start_frame =  cfg.DATASETS.START_FRAME
        self.num_frame =  cfg.DATASETS.NUM_FRAME
        self.use_depth_proxy = cfg.DATASETS.USE_DEPTH_PROXY
        self.use_campose = cfg.DATASETS.USE_CAMPOSE
        self.use_bg = cfg.DATASETS.USE_BACKGROUND
        self.use_skeleton = cfg.DATASETS.USE_SKELETON
        self.mask_sampling = cfg.DATASETS.MASK_SAMPLING
        self.skip = cfg.DATASETS.SKIP
        self.size = (cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0])

        #NHR datasets
        self.NHR_dataset = IBRDynamicDataset(data_folder_path, self.start_frame, self.num_frame, False, self.use_bg, 
                                            transforms, [1.0, 6.5, 0.8], skip_step = self.skip, random_noisy = 0, holes='None')
        
        # select cams
        self.num_cam = self.NHR_dataset.cam_num
        self.selected_cams = self.NHR_dataset.selected_cams

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

    def __len__(self):
        return len(self.NHR_dataset)

    def get_bbox(self, vs, epd = 0.1):
        max_xyz = torch.max(vs, dim=0)[0]
        min_xyz = torch.min(vs, dim=0)[0]

        tmp = (max_xyz - min_xyz) * epd

        max_xyz = max_xyz + tmp
        min_xyz = min_xyz - tmp

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        return torch.from_numpy(bbox.astype(np.float32)).reshape((1, 8, 3))

    def sample_ray(self, frame_id, cam_id):

        img = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/img_{cam_id:04d}.jpg'))
        #build rays
        T = self.Ts[cam_id].clone()
        K = self.Ks[cam_id].clone()
        K[:2,:] = K[:2,:] * self.size[1] / img.size[0]   

        img = TF.functional.to_tensor(img)
        img = TF.functional.resize(img, self.size)

        rays, rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = torch.ones(img[0,:,:].size()).unsqueeze(0), mask_threshold = 0.5, images = img.unsqueeze(0))

        mask = Image.open(os.path.join(self.data_folder_path,f'img/{frame_id}/mask_{cam_id:04d}.jpg'))
        mask = TF.functional.to_tensor(mask)
        mask = TF.functional.resize(mask, self.size)
        mask = mask[:1,:,:].reshape(-1,1)
        mask[mask > 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        n_body = int(self.bunch * 0.5)
        n_rand = self.bunch - n_body 

        # sample rays on body
        coord_body = np.argwhere(mask.numpy() > 0.5)[:,0]
     
        coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                    n_body)]
                                            
        # sample rays in the bound mask
        coord = np.argwhere(mask.numpy() < 0.5)[:,0]
        coord = coord[np.random.randint(0, len(coord), n_rand)]

        coord = np.concatenate([coord_body, coord], axis=0)

        rays = rays[coord]
        rgbs = rgbs[coord]
        mask = mask[coord]

        if self.use_depth_proxy:
            depth = torch.load(os.path.join(self.data_folder_path,f'depth/{frame_id}/depth_{cam_id:04d}.pt')).float()
            depth = TF.functional.resize(depth.unsqueeze(0), self.size)
            depth = depth.reshape(-1,1)[coord]
        else:
            depth = None


        if self.use_bg:
            bg = Image.open(os.path.join(self.data_folder_path,f'img/background/img_{cam_id:04d}.jpg'))
            bg = TF.functional.to_tensor(bg)
            bg = TF.functional.resize(bg, self.size)
            bg = bg.reshape(-1,3)[coord]
        else:
            bg = None

        return rays, rgbs, mask, depth ,bg

    '''
    output:

    rays: (N,6)
    rgbs:(N,3)
    bbox:(N,8,3)
    near_fars: (N,2)
    frame_ids: (N,1)
    '''
    def __getitem__(self, index):
        sample = {}

        frame_id = ((index // self.num_cam) * self.skip) % self.num_frame 
        cam_id = self.selected_cams[index % self.num_cam]

        #build rays
        rays, rgbs, mask, depth ,bg = self.sample_ray(frame_id, cam_id)

        frame_ids = torch.ones((self.bunch,1))* frame_id
        cam_ids = torch.ones((self.bunch,1))* cam_id

        sample['rays'] = rays
        sample['rgbs'] = rgbs
        sample['mask'] = mask

        sample['near_fars'] = torch.tensor([self.NHR_dataset.near[cam_id],self.NHR_dataset.far[cam_id]]).unsqueeze(0).repeat(self.bunch, 1)

        sample['vertices'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, f'pointclouds/frame{frame_id + 1}.npy')).astype(np.float32))
        sample['bboxes'] =  self.get_bbox(sample['vertices']).repeat(self.bunch, 1, 1)

        sample['frame_ids'] = frame_ids    
        sample['cam_ids'] = cam_ids   

        if self.use_depth_proxy:
            sample['depth'] = depth
    
        if self.use_bg:
            sample['bgs'] = bg


        if self.use_skeleton:
            sample['Rts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/Rt{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1,1) 
            sample['global_ts'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/global_t{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1) 
            sample['skeletons'] = torch.from_numpy(np.load(os.path.join(self.data_folder_path, 
                                                                f'skeletons/skeletonpose{frame_id}.npy')).astype(np.float32)).unsqueeze(0).repeat(self.bunch,1,1) 
        return sample