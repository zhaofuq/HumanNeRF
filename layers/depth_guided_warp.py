import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

class LaplacianKernel(nn.Module):
    def __init__(self):
        super(LaplacianKernel, self).__init__()
        kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)/4.0

    def forward(self, x):
        self.kernel = self.kernel.to(x.device)
        return F.conv2d(x, self.kernel, padding=1)

def body_weights_sample(weights, vertices, xyz):

    ret = knn_points(xyz.unsqueeze(0), vertices.unsqueeze(0), None, None, 1)
    dist, idx = ret[0].squeeze(-1) ,ret[1].squeeze(-1)
    weight = weights[idx]

    return weight

def bilinear_sample(xyz, color):
    '''
    :param xyz:
    :param color: B*H*W*3
    :return: color, mask
    '''

    b, h, w = color.shape[:3]

    xx = xyz[:, :, 0]
    yy = xyz[:, :, 1]

    xx = torch.min(xx, ( w -1 ) *torch.ones_like(xx, device=color.device))
    xx = torch.max(xx, torch.zeros_like(xx, device=color.device))

    yy = torch.max(yy, torch.zeros_like(yy, device=color.device))
    yy = torch.min(yy, ( h -1 ) *torch.ones_like(yy, device=color.device))

    xx_floor = torch.floor(xx).long()
    yy_floor = torch.floor(yy).long()
    xx_ceil = xx_floor + 1
    yy_ceil = yy_floor + 1

    xx_ceil = torch.min(xx_ceil, ( w -1 ) *torch.ones_like(xx_ceil, device=color.device))
    xx_ceil = torch.max(xx_ceil, torch.zeros_like(xx_ceil, device=color.device))

    xx_floor = torch.min(xx_floor, (w - 1) * torch.ones_like(xx_floor, device=color.device))
    xx_floor = torch.max(xx_floor, torch.zeros_like(xx_floor, device=color.device))

    yy_ceil = torch.max(yy_ceil, torch.zeros_like(yy_ceil, device=color.device))
    yy_ceil = torch.min(yy_ceil, ( h -1 ) *torch.ones_like(yy_ceil, device=color.device))

    yy_floor = torch.max(yy_floor, torch.zeros_like(yy_floor, device=color.device))
    yy_floor = torch.min(yy_floor, ( h -1 ) *torch.ones_like(yy_floor, device=color.device))

    batch = torch.arange(b, device=color.device).reshape((b, 1)).repeat((1, h* w))
    Ia = color[batch, yy_floor, xx_floor]
    Ib = color[batch, yy_ceil, xx_floor]
    Ic = color[batch, yy_floor, xx_ceil]
    Id = color[batch, yy_ceil, xx_ceil]
    xx_floor = xx_floor.float()
    xx_ceil = xx_ceil.float()
    yy_floor = yy_floor.float()
    yy_ceil = yy_ceil.float()

    w3 = ((xx - xx_floor) * (yy - yy_floor)).unsqueeze(2)
    w2 = ((xx - xx_floor) * (yy_ceil - yy)).unsqueeze(2)
    w1 = ((xx_ceil - xx) * (yy - yy_floor)).unsqueeze(2)
    w0 = ((xx_ceil - xx) * (yy_ceil - yy)).unsqueeze(2)

    color_pred = w0 * Ia + w1 * Ib + w2 * Ic + w3 * Id

    # print("max w", torch.max(w0+w1+w2+w3))
    # color_pred = color[batch, yy_floor, xx_ceil]

    # print("color_pred ", torch.max(color_pred))
    # print("color ", torch.max(color))
    return color_pred.reshape((b, h, w, 3))


def bilinear_sample_function(xyz, color, padding_mode):
    b, c, h, w = color.shape[:4]

    xx = xyz[:, :, 0].unsqueeze(-1)
    yy = xyz[:, :, 1].unsqueeze(-1)

    xx_norm = xx / (w - 1) * 2 - 1
    yy_norm = yy / (h - 1) * 2 - 1
    xx_mask = ((xx_norm > 1) + (xx_norm < -1)).detach()
    yy_mask = ((yy_norm > 1) + (yy_norm < -1)).detach()

    if padding_mode == 'zeros':
        xx_norm[xx_mask] = 2 
        yy_norm[yy_mask] = 2
    # mask = ((xx_norm > 1) + (xx_norm < -1) + (yy_norm < -1) + (yy_norm > 1)).detach().squeeze()
    # mask = mask.unsqueeze(1).expand(b, 3, h * w)

    pixel_coords = torch.stack([xx_norm, yy_norm], dim=2).reshape((b, h, w, 2))  # [B, H*W, 2]
    color_pred = F.grid_sample(color, pixel_coords, padding_mode=padding_mode, align_corners=True)
    return color_pred


def warp(depth_tgt, color_src, intrinc_src, intrinc_tgt, extrinc_src, extrinc_tgt):
    '''
    :param depth_tgt: B*1*H*W
    :param color_src: B*3*H*W
    :param intrinc_src: B*3*3
    :param extrinc_src: B*4*4
    :param extrinc_tgt: B*4*4
    :return: color_tgt
        mask_tgt
    '''
    depth_tgt = depth_tgt.permute(0, 2, 3, 1).contiguous()
    b, h, w = depth_tgt.shape[:3]

    x = torch.arange(0, w, device=depth_tgt.device).float()
    y = torch.arange(0, h, device=depth_tgt.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth_tgt.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))
    dd = depth_tgt.reshape((b, h * w, 1))

    center = torch.cat((intrinc_tgt[:, 0, 2].unsqueeze(1), intrinc_tgt[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc_tgt[:, 0, 0].unsqueeze(1), intrinc_tgt[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz = (xyz * dd).transpose(1, 2)

#     R_src_inv = (extrinc_src[:, :3, :3]).transpose(1, 2).reshape((b, 3, 3))
#     R_tgt = extrinc_tgt[:, :3, :3].reshape((b, 3, 3))
#     t_src = extrinc_src[:, :3, 3].reshape((b, 3, 1))
#     t_tgt = extrinc_tgt[:, :3, 3].reshape((b, 3, 1))
#     R12 = torch.matmul(R_src_inv, R_tgt)
#     t12 = torch.matmul(R_src_inv, t_tgt) - torch.matmul(R_src_inv, t_src)
    
    RT = torch.matmul(extrinc_src.inverse(), extrinc_tgt)
    R12 = RT[:,:3,:3]
    t12 = RT[:,:3,3]
    xyz = torch.matmul(R12, xyz) + t12.reshape((-1,3,1))
    
    xyz = torch.matmul(intrinc_src, xyz)
    xyz = (xyz / (xyz[:, 2, :].unsqueeze(1) + np.finfo(float).eps)).transpose(1, 2)
    
    color_tgt = bilinear_sample_function(xyz, color_src,  'border')
    
    return color_tgt

def warp_pose(depth_tgt, color_src, intrinc_src, intrinc_tgt, extrinc_src,extrinc_tgt,
        Rts_src,  Rts_tgt, global_ts_src, global_ts_tgt, skinning_weights, vertices):
    '''
    :param depth_tgt: B*1*H*W
    :param color_src: B*3*H*W
    :param intrinc_src: B*3*3
    :param extrinc_src: B*4*4
    :param extrinc_tgt: B*4*4
    :return: color_tgt
        mask_tgt
    '''
    depth_tgt = depth_tgt.permute(0, 2, 3, 1).contiguous()
    b, h, w = depth_tgt.shape[:3]

    #2d coord
    x = torch.arange(0, w, device=depth_tgt.device).float()
    y = torch.arange(0, h, device=depth_tgt.device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = (xx.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    yy = (yy.unsqueeze(0).repeat((b, 1, 1))).unsqueeze(-1)
    ones_tensor = torch.ones_like(xx, device=depth_tgt.device)
    xyz = torch.cat((xx, yy, ones_tensor), dim=3).reshape((b, h * w, 3))

    dd = depth_tgt.reshape((b, h * w, 1))
    mask = dd > 0.5

    center = torch.cat((intrinc_tgt[:, 0, 2].unsqueeze(1), intrinc_tgt[:, 1, 2].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    focal = torch.cat((intrinc_tgt[:, 0, 0].unsqueeze(1), intrinc_tgt[:, 1, 1].unsqueeze(1)), dim=1).unsqueeze(
        1).reshape((-1, 1, 2))
    xyz[:, :, :2] = (xyz[:, :, :2] - center) / focal
    xyz_mask = (xyz[mask.squeeze(-1),:] * dd[mask.squeeze(-1),:]).unsqueeze(0).transpose(1, 2)

    #cam2world
    R12 = extrinc_tgt[:,:3,:3]
    t12 = extrinc_tgt[:,:3,3]
    xyz_mask = torch.matmul(R12, xyz_mask) + t12.reshape((-1,3,1))
    xyz_mask = xyz_mask[0].permute(1,0)

    #PSOEA->POSEB

    W = body_weights_sample(skinning_weights, vertices, xyz_mask)

    wRts_src = torch.matmul(W, Rts_src).reshape(-1, 4, 4)
    wRts_tgt = torch.matmul(W, Rts_tgt).reshape(-1, 4, 4)
    wRts = torch.matmul(wRts_src, wRts_tgt.inverse())

    xyz_mask = torch.sum((xyz_mask - global_ts_tgt)[:,None,:] * wRts[:,:3,:3], dim=-1) + wRts[:,:3,3] + global_ts_src

    #world2cam
    RT = extrinc_src.inverse()#extrinc_tgt
    R12 = RT[:,:3,:3]
    t12 = RT[:,:3,3]
    xyz_mask = torch.matmul(R12, xyz_mask.unsqueeze(0).transpose(1,2)) + t12.reshape((-1,3,1))

    #projection
    xyz_mask = torch.matmul(intrinc_src, xyz_mask)
    xyz_mask = (xyz_mask / (xyz_mask[:, 2, :].unsqueeze(1) + np.finfo(float).eps)).transpose(1, 2)

    xyz_pixel = torch.zeros(b, h* w, 3)
    xyz_pixel[mask.squeeze(-1),:] = xyz_mask
 
    color_tgt = bilinear_sample_function(xyz_pixel, color_src,  'border')
    
    return color_tgt


if __name__ == '__main__':
    import cv2
    import numpy as np

    path = '/data/new_disk/suoxin/RealTimeDeepHumanData/data6/'
    color_src = cv2.imread(path + '/rgb/image_30.bmp')
    print(color_src.dtype)

    color_tgt = cv2.imread(path + '/rgb/image_29.bmp')
    depth_src = (cv2.imread(path + '/depth/depth_30.bmp')).astype(np.int)
    depth_tgt = (cv2.imread(path + '/depth/depth_29.bmp')).astype(np.int)

    extrinc_src = np.linalg.inv(np.loadtxt(path + '/extrinc/extrinc_30.inf'))
    extrinc_tgt = np.linalg.inv(np.loadtxt(path + '/extrinc/extrinc_29.inf'))

    mask_src = np.ones_like(depth_src[:, :, 0])
    mask_tgt = np.ones_like(depth_src[:, :, 0])
    mask_src[depth_src[:, :, 0] == depth_src[0, 0, 0]] = 0
    mask_tgt[depth_tgt[:, :, 0] == depth_tgt[0, 0, 0]] = 0
    mask_src = mask_src.astype(np.int)
    mask_tgt = mask_tgt.astype(np.int)
    dd_src = (depth_src[:, :, 2].astype(np.int) + \
              ((depth_src[:, :, 1].astype(np.int)) << 8) + \
              ((depth_src[:, :, 0].astype(np.int)) << 16)) / 65536.0

    dd_tgt = (depth_tgt[:, :, 2].astype(np.int) + \
              ((depth_tgt[:, :, 1]) << 8).astype(np.int) + \
              ((depth_tgt[:, :, 0]) << 16).astype(np.int)) / 65536.0

    dd_src = dd_src * mask_src

    dd_tgt = dd_tgt * mask_tgt

    intrinc = np.array([[309.019, 0, 128], [0, 309.019, 128], [0, 0, 1]])

    depth_tgt = torch.from_numpy(dd_tgt).unsqueeze(0)

    color_src = torch.from_numpy(color_src).unsqueeze(0)
    intrinc_src = torch.from_numpy(intrinc).unsqueeze(0)
    intrinc_tgt = torch.from_numpy(intrinc).unsqueeze(0)
    extrinc_src = torch.from_numpy(extrinc_src).unsqueeze(0)
    extrinc_tgt = torch.from_numpy(extrinc_tgt).unsqueeze(0)

    depth_tgt = depth_tgt.unsqueeze(1)
    color_src = color_src.permute(0, 3, 1, 2)
    print("depth max", torch.max(depth_tgt))
    image = warp(depth_tgt.float(), color_src.float(), intrinc_src.float(), intrinc_tgt.float(),
                       extrinc_src.float(), extrinc_tgt.float())

    image = image.permute(0, 2, 3, 1)[0].cpu().numpy()
    cv2.imwrite("./image.png", image.astype(np.uint8))
    
    cv2.imwrite("./color_tgt.png", color_tgt.astype(np.uint8))
    
    # cv2.imshow("win ", image.astype(np.uint8))
    # cv2.waitKey()
    print("image ", image.shape)