
import torch


'''
Sample rays from views (and images) with/without masks

--------------------------
INPUT Tensors
Ks: intrinsics of cameras (M,3,3)
Ts: extrinsic of cameras (M,4,4)
image_size: the size of image [H,W]
images: (M,C,H,W)
mask_threshold: a float threshold to mask rays
masks:(M,H,W)
-------------------
OUPUT:
list of rays:  (N,6)  dirs(3) + pos(3)
RGB:  (N,C)
'''

def ray_sampling(Ks, Ts, image_size, masks=None, mask_threshold = 0.5, images=None):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)


    x = torch.linspace(0,h-1,steps=h,device = Ks.device )
    y = torch.linspace(0,w-1,steps=w,device = Ks.device )

    grid_x, grid_y = torch.meshgrid(x,y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M,1,1,1)   #(M,2,H,W)
    coordinates = torch.cat([coordinates,torch.ones(coordinates.size(0),1,coordinates.size(2), 
                             coordinates.size(3),device = Ks.device) ],dim=1).permute(0,2,3,1).unsqueeze(-1)


    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks,coordinates) #(M,H,W,3,1)

    #dirs = dirs/torch.norm(dirs,dim=3,keepdim = True)
    
    dirs = torch.cat([dirs,torch.zeros(dirs.size(0),coordinates.size(1), 
                             coordinates.size(2),1,1,device = Ks.device) ],dim=3) #(M,H,W,4,1)


    dirs = torch.matmul(Ts,dirs) #(M,H,W,4,1)
    dirs = dirs[:,:,:,0:3,0]  #(M,H,W,3)

    pos = Ts[:,0:3,3] #(M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1,h,w,1)

    rays = torch.cat([dirs,pos],dim = 3)  #(M,H,W,6)

    if images is not None:
        rgbs = images.permute(0,2,3,1) #(M,H,W,C)
    else:
        rgbs = None

    if masks is not None:
        rays = rays[masks>mask_threshold,:]
        if rgbs is not None:
            rgbs = rgbs[masks>mask_threshold,:]

    else:
        rays = rays.reshape((-1,rays.size(3)))
        if rgbs is not None:
            rgbs = rgbs.reshape((-1, rgbs.size(3)))

    return rays,rgbs
    






