import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

import time

def merge_holes(pc1,pc2):

    # change point color here

    return np.concatenate([pc1, pc2], axis=0)


class IBRDynamicDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_path, start_frame, frame_num, use_mask, use_bg, transforms, near_far_size, skip_step, random_noisy, holes):
        super(IBRDynamicDataset, self).__init__()
        self.start_frame = start_frame
        self.frame_num = frame_num
        self.data_folder_path = data_folder_path
        self.use_mask = use_mask
        self.use_bg = use_bg
        self.skip_step = skip_step
        self.random_noisy  =random_noisy
        self.holes = holes

        self.file_path = os.path.join(data_folder_path,'img')
        self.use_pointclouds = True

        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index =[]

        sum_tmp = 0
        for i in range(start_frame, start_frame + frame_num):
            #tmp = np.loadtxt(os.path.join(data_folder_path,'pointclouds/frame%d.obj' % (i+1)), usecols = (1,2,3,4,5,6))
            if not os.path.exists(os.path.join(data_folder_path,'pointclouds/frame%d.npy' % (i+1))):
                break

            tmp = np.load(os.path.join(data_folder_path,'pointclouds/frame%d.npy' % (i+1)))
            
            if os.path.exists(os.path.join(self.holes,'holes/frame%d.npy' % (i+1))):
                tmp2 = np.load(os.path.join(self.holes,'holes/frame%d.npy' % (i+1)))
                tmp = merge_holes(tmp, tmp2)
                if i%50 == 0:
                    print('merge holes', tmp2.shape[0])


            vs_tmp = tmp[:,0:3] 
            vs_rgb_tmp = tmp[:,3:6]
            self.vs_index.append(sum_tmp)
            self.vs.append(torch.Tensor(vs_tmp))
            self.vs_rgb.append(torch.Tensor(vs_rgb_tmp))
            self.vs_num.append(vs_tmp.shape[0])
            sum_tmp = sum_tmp + vs_tmp.shape[0]



        if len(self.vs)>0:
            self.vs = torch.cat( self.vs, dim=0 )
            self.vs_rgb = torch.cat( self.vs_rgb, dim=0 )

        if random_noisy>0:
            n = tdist.Normal(torch.tensor([0.0, 0.0,0.0]), torch.tensor([random_noisy,random_noisy,random_noisy]))
            kk = torch.min((torch.max(self.vs,dim = 1)[0] - torch.min(self.vs,dim = 1)[0])/500)
            self.vs = self.vs + kk*n.sample((self.vs.size(0),))
        
        

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

        if os.path.exists(os.path.join(data_folder_path,'CamSelected.inf')):
            self.selected_cams = list(np.loadtxt(os.path.join(data_folder_path,'CamSelected.inf')).astype(np.uint8))
            self.cam_num = len(self.selected_cams)
        else:
            self.cam_num = self.Ts.size(0)
            self.selected_cams = [i for i in range(self.cam_num)]
        

        '''
        for i in range(self.Ks.size(0)):
            if self.Ks[i,0,2] > 1100:
                self.Ks[i] = self.Ks[i] * 2048.0/2448.0
                self.Ks[i] = self.Ks[i] / (2048.0/800)
            else:
                self.Ks[i] = self.Ks[i] / (2048.0/800)

        self.Ks[:,2,2] = 1
        '''

        self.transforms = transforms
        self.near_far_size = torch.Tensor(near_far_size)

        #self.black_list = [625,747,745,738,62,750,746,737,739,762]

        if type(self.vs)==list:
            self.use_pointclouds = False
            self.vs = self.Ts[:,0:3,3].repeat(frame_num, 1)
            self.vs_rgb = torch.zeros_like(self.vs).repeat(frame_num, 1)
            for i in range(frame_num):
                self.vs_num.append(self.Ts.shape[0])
                self.vs_index.append(i * self.Ts.shape[0])



        print('load %d Ts, %d Ks, %d frame, %d vertices' % (self.cam_num,self.cam_num,self.frame_num,self.vs.size(0)))


        self._all_imgs = None
        self._all_Ts = None
        self._all_Ks = None
        self._all_width_height = None

        inv_Ts = torch.inverse(self.Ts).unsqueeze(1)  #(M,1,4,4)
        vs = self.vs.clone().unsqueeze(-1)   #(N,3,1)
        vs = torch.cat([vs,torch.ones(vs.size(0),1,vs.size(2)) ],dim=1) #(N,4,1)

        pts = torch.matmul(inv_Ts,vs) #(M,N,4,1)

        pts_max = torch.max(pts, dim=1)[0].squeeze() #(M,4)
        pts_min = torch.min(pts, dim=1)[0].squeeze() #(M,4)

        pts_max = pts_max[:,2]   #(M)
        pts_min = pts_min[:,2]   #(M)
        
        if self.use_pointclouds:
            self.near = pts_min *0.5
            self.near[self.near<(pts_max*0.1)] = pts_max[self.near<(pts_max*0.1)]*0.1
            
            self.far = pts_max *2
        else:
            self.near = pts_min
            self.far = pts_max

        print('dataset initialed. near: %f  far: %f'%(self.near.min(),self.far.max()))
        

    def __len__(self):
        return self.cam_num *  (self.frame_num // self.skip_step) 

    def __getitem__(self, index, need_transform = True):


        frame_id = ((index // self.cam_num) * self.skip_step) %self.frame_num
        cam_id = self.selected_cams[index % self.cam_num]
        
        start = time.time()
        img = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id + self.start_frame, cam_id)))


        K = self.Ks[cam_id]

        if self.use_mask:
            img_mask = Image.open(os.path.join(self.file_path,'%d/mask/img_%04d.jpg' % ( frame_id + self.start_frame, cam_id)))

            img, K, T, img_mask, ROI = self.transforms(img,self.Ks[cam_id],self.Ts[cam_id],img_mask)
            img = torch.cat([img,img_mask[0:1,:,:]], dim=0)
        else:
            
            img,K,T,_,ROI = self.transforms(img,self.Ks[cam_id],self.Ts[cam_id])
            img = torch.cat([img, torch.ones(img[0:1,:,:].size())], dim=0)

        img = torch.cat([img,ROI], dim=0)

        if self.use_bg:
            bg = Image.open(os.path.join(self.file_path,'background/img_%04d.jpg' % cam_id))
            bg,_,_,_,ROI = self.transforms(bg,self.Ks[cam_id],self.Ts[cam_id])
            img = torch.cat([img,bg], dim=0)
        else:
            img = torch.cat([img, torch.zeros(img[0:3,:,:].size())], dim=0)

        return img, self.vs[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:], frame_id, T, K, torch.tensor([self.near[cam_id],self.far[cam_id]]).unsqueeze(0), self.vs_rgb[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:],cam_id

    def get_vertex_num(self):
        return torch.Tensor(self.vs_num)