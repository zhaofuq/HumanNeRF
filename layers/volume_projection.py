import numpy as np
import torch
from torch import nn

def volume_dependview(coordinate, dmin, dmax, num):
    
    b, n = coordinate.shape[:2]
    grid_depth = torch.range(0, num-1).reshape((1,1,num, 1)).repeat((b, 1, 1, 1)) * (dmax-dmin)/num + dmin # b * m
    grid_depth = grid_depth.view(b,1,num,1)
    coordinate = coordinate.view(b,n,1,3)
    grid_volume = coordinate * grid_depth # b*n*m*3
    return grid_volume.reshape(b, -1, 3), grid_depth

# b * n * m * 1
def volume_dependview_adaptive(coordinate, dmin, dmax, num):
    b, n = coordinate.shape[:2]
    dmin = dmin.view((b, -1, 1, 1))
    dmax = dmax.view((b, -1, 1, 1))
    grid_depth = torch.range(0, num-1).reshape((1,1,num, 1)).to(coordinate.device).repeat((b, 1, 1, 1)) * (dmax-dmin)/num + dmin # b * n * m * 1
    
    coordinate = coordinate.view(b,n,1,3)
    grid_volume = coordinate * grid_depth # b*n*m*3
    
    return grid_volume.reshape(b, -1, 3), grid_depth
    
# occupancy_grid b*h*w*d
# def project_occupancy_depth(occupancy_grid, dmin, dmax, num):
#     d = occupancy_grid.shape[3]-1
#     occupancy_grid[occupancy_grid<0.6]=0
#     occupancy_grid[occupancy_grid>=0.6]=1
#     occupancy_grid = occupancy_grid - torch.arange(d).view(1,1,1,d).to(occupancy_grid.device)*0.00001
    
#     indices = torch.argmax(occupancy_grid, dim=3, keepdim=True) # b*h*w*1
#     depth = (dmax-dmin)*indices/num + dmin
#     return depth
    
def project_occupancy_depth(occupancy_grid, volume_depth):
    b, h, w, d = occupancy_grid.shape
    volume_depth = volume_depth.view(b,h,w,d)
    d = occupancy_grid.shape[3]
    occupancy_grid[occupancy_grid<=0.5]=0.0
    occupancy_grid[occupancy_grid>0.5]=1.0
    
    occupancy_grid = occupancy_grid - torch.arange(d).view(1,1,1,d).to(occupancy_grid.device)*0.00001
    
    indices = torch.argmax(occupancy_grid, dim=3, keepdim=True) # b*h*w*1
    pre_index = indices-1
    pre_index[pre_index<0]=0
    pre_depth = torch.gather(volume_depth, dim=3, index = pre_index.long())
    depth = torch.gather(volume_depth, dim=3, index = indices.long())
    return (depth + pre_depth)/2.0 


def gen_volume_dependview(intrinc, extrinc, dmin_tensor, dmax_tensor, num):
    b,h,w = dmin_tensor.shape[:3]
    x = torch.arange(w).float()
    y = torch.arange(h).float()
    ys, xs = torch.meshgrid(y, x)
    ys = ys.contiguous().view(h*w, 1)
    xs = xs.contiguous().view(h*w, 1)
    points3d = torch.cat((xs, ys, torch.ones_like(xs)), dim=1).reshape((-1, 3)).to(intrinc.device)
    points3d_dir = torch.matmul(torch.inverse(intrinc), points3d.transpose(0, 1))# normalized coordinate b*3*n(h*w)
    # points3d_dir * depth = (x, y, d)
    sample_points, volume_depth = volume_dependview_adaptive(points3d_dir.transpose(1,2), dmin_tensor, dmax_tensor, num)
    sample_points = (torch.matmul(extrinc[:,:3,:3], sample_points.transpose(1, 2)) + extrinc[:,:3,3].view((-1, 3, 1))).transpose(1, 2)
    return sample_points, volume_depth

def gen_volume_dependview1(points3d, intrinc, extrinc, dmin_tensor, dmax_tensor, num):

    points3d_dir = torch.matmul(torch.inverse(intrinc), points3d.transpose(0, 1))

    sample_points, volume_depth = volume_dependview_adaptive(points3d_dir.transpose(1,2), dmin_tensor, dmax_tensor, num)
    sample_points = (torch.matmul(extrinc[:,:3,:3], sample_points.transpose(1, 2)) + extrinc[:,:3,3].view((-1, 3, 1))).transpose(1, 2)
    return sample_points, volume_depth



class VolumeDepthProjection(nn.Module):
    def __init__(self, cfg, h=256, w=256):
        super().__init__()
        self.cfg = cfg
        x = torch.arange(w).float()
        y = torch.arange(h).float()
        ys, xs = torch.meshgrid(y, x)
        ys = ys.contiguous().view(h*w, 1)
        xs = xs.contiguous().view(h*w, 1)
        points3d = torch.cat((xs, ys, torch.ones_like(xs)), dim=1).reshape((-1, 3))
        
        self.register_buffer("points3d", points3d)
        self.h= h
        self.w =w
        
    def gen_volume_dependview(self, intrinc, extrinc, dmin_tensor, dmax_tensor, num):
        
        points3d_dir = torch.matmul(torch.inverse(intrinc), self.points3d.transpose(0, 1).to(intrinc.device))
        
        
        sample_points, volume_depth = volume_dependview_adaptive(points3d_dir.transpose(1,2), dmin_tensor, dmax_tensor, num)
        sample_points = (torch.matmul(extrinc[:,:3,:3], sample_points.transpose(1, 2)) + extrinc[:,:3,3].view((-1, 3, 1))).transpose(1, 2)
        
        return sample_points, volume_depth, points3d_dir.view((-1, 3, self.h, self.w))
    


