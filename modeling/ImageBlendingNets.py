from torch import nn
import functools
import torch
import torch.nn.functional as F


class UpConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding_mode  = 'replicate',  padding=1),
                                  nn.ReLU())
        
    
    def forward(self, inputs):
        b,c,h,w = inputs.shape
        upsample = F.upsample(inputs, size = [h*2, w*2], mode = 'bilinear')
        
        return self.conv(upsample)
    
class DownConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DownConv, self).__init__()
        self.downconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding_mode  = 'replicate',  padding=1), nn.ReLU(),
                                     nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding_mode  = 'replicate',  padding=1), nn.ReLU())
        
    
    def forward(self, inputs):
        return self.downconv(inputs)

class ImageBlending1(nn.Module):
    def __init__(self, cfg, ch_in):
        super(ImageBlending1, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, 32, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1), nn.ReLU())
        self.conv2 = DownConv(32, 48,3, 2)
        self.conv3 = DownConv(48, 64,3, 2)
        self.conv4 = DownConv(64, 96,3, 2)
        self.conv5 = DownConv(96, 128,3, 2)
        self.conv6 = UpConv(128, 96, 3)
        
        self.conv7 = UpConv(96*2, 64, 3)
        self.conv8 = UpConv(64*2, 48, 3)
        self.conv9 = UpConv(48*2, 32, 3)
        
        self.conv = nn.Sequential(nn.Conv2d(32, 2, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1))
        
        
    def forward(self, image1, image2, occlusion=None):
        b, c, h, w = image1.shape
        if(occlusion is not None):
            images = torch.cat([image1, image2, occlusion], dim=1)
        else:
            images = torch.cat([image1, image2], dim=1)
#         print('images ', images.shape)
        x1 = self.conv1(images)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x4], dim=1))
        x8 = self.conv8(torch.cat([x7, x3], dim=1))
        x9 = self.conv9(torch.cat([x8, x2], dim=1))
        x = self.conv(x9)
        weight = F.softmax(x[:,:2], dim=1)

        image = weight[:,0,:,:].unsqueeze(1) * image1[:,:3] + weight[:,1,:,:].unsqueeze(1) * image2[:,:3]
        return image, weight

class ImageBlending2(nn.Module):
    def __init__(self, cfg, ch_in, ch_out=3):
        super(ImageBlending2, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, 32, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1), nn.ReLU())
        self.conv2 = DownConv(32, 48,3, 2)
        self.conv3 = DownConv(48, 64,3, 2)
        self.conv4 = DownConv(64, 96,3, 2)
        self.conv5 = DownConv(96, 128,3, 2)
        self.conv6 = UpConv(128, 96, 3)
        
        self.conv7 = UpConv(96*2, 64, 3)
        self.conv8 = UpConv(64*2, 48, 3)
        self.conv9 = UpConv(48*2, 32, 3)
        
        self.conv = nn.Sequential(nn.Conv2d(32, ch_out, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1))
        
        
    def forward(self, image1, image2, render=None):
        b, c, h, w = image1.shape
        if(render is not None):
            images = torch.cat([image1, image2, render[:,3:]], dim=1)
        else:
            images = torch.cat([image1, image2], dim=1)

        x1 = self.conv1(images)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x4], dim=1))
        x8 = self.conv8(torch.cat([x7, x3], dim=1))
        x9 = self.conv9(torch.cat([x8, x2], dim=1))
        x = self.conv(x9)
        weight = F.softmax(x[:,:3], dim=1)

        image = weight[:,0,:,:].unsqueeze(1) * image1[:,:3] + weight[:,1,:,:].unsqueeze(1) * image2[:,:3] + weight[:,2,:,:].unsqueeze(1) * render[:,:3]
        
        return image, weight

class ImageBlending3(nn.Module):
    def __init__(self, cfg, ch_in,ch_out=3):
        super(ImageBlending3, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, 32, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1), nn.ReLU())
        self.conv2 = DownConv(32, 48,3, 2)
        self.conv3 = DownConv(48, 64,3, 2)
        self.conv4 = DownConv(64, 96,3, 2)
        self.conv5 = DownConv(96, 128,3, 2)
        self.conv6 = UpConv(128, 96, 3)
        
        self.conv7 = UpConv(96*2, 64, 3)
        self.conv8 = UpConv(64*2, 48, 3)
        self.conv9 = UpConv(48*2, 32, 3)
        
        self.conv = nn.Sequential(nn.Conv2d(32, ch_out, kernel_size = 3, stride = 1, padding_mode  = 'replicate', padding=1))
        
        
    def forward(self, image1, image2, render=None):
        b, c, h, w = image1.shape
        if(render is not None):
            images = torch.cat([image1, image2, render], dim=1)
        else:
            images = torch.cat([image1, image2], dim=1)

        x1 = self.conv1(images)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x4], dim=1))
        x8 = self.conv8(torch.cat([x7, x3], dim=1))
        x9 = self.conv9(torch.cat([x8, x2], dim=1))
        x = self.conv(x9)
        image = torch.sigmoid(x[:,:3])
        
        return image
