import torch.nn as nn
import torch

class DepthRegress(nn.Module):
    def __init__ (self):
        super(DepthRegress, self).__init__()
        # self.cfg = cfg
        channels_out = 1
        self.conv1 = nn.Conv2d(65, 65, 4, 2, 1)
        self.conv2 = nn.Conv2d(65, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.dconv3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.dconv2 = nn.Conv2d(256, 64, 3, 1, 1)
        self.dconv1 = nn.Conv2d(129, 64, 3, 1, 1)
        self.dconv = nn.Conv2d(64, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        
        e1 = self.conv1(input) # 128 * 128
        e2 = (self.conv2(self.leaky_relu(e1)))# 64 * 64 * 128
        e3 = self.conv3(self.leaky_relu(e2))# 32 * 32
        d3 = self.dconv3(self.up(self.relu(e3))) # 128
        d3 = torch.cat((d3, e2), 1)  # 256
        d2 = self.dconv2(self.up(self.relu(d3)))
        d2 = torch.cat((d2, e1), 1)  # 128 128
        d1 = self.dconv1(self.up(self.relu(d2)))
        d1 = self.dconv(self.relu(d1))
        depth = self.tanh(d1)
        return depth
    
if __name__ == '__main__':
    dr = DepthRegress()
    input = torch.zeros((8, 65, 256, 256))
    print(dr(input).shape)
    