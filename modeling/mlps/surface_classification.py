
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=6, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
                
            if self.num_views > 1 and i == 1:
                y = (y.view(-1, self.num_views, y.shape[1], y.shape[2])).mean(1)
                tmpy = (feature.view(-1, self.num_views, feature.shape[1], feature.shape[2])).mean(1)
            
        if self.last_op:
            y = self.last_op(y)
        return y
    
    
class WeightSurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=6, no_residual=True, last_op=None):
        super(WeightSurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])
        
        self.weight_net = nn.Conv1d(filter_channels[2], filter_channels[2], 1)
        self.weight_net1 = nn.Conv1d(filter_channels[0], filter_channels[0], 1)
        
        

    def forward(self, feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
                
            if self.num_views > 1 and i == 1:
                
                weights = self.weight_net(y).view(-1, self.num_views, y.shape[1], y.shape[2])
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2])
                weights = torch.softmax(weights, dim=1)
                y = (weights *y ).sum(dim=1)
                
                tmp_weights = self.weight_net1(tmpy).view(-1, self.num_views, tmpy.shape[1], tmpy.shape[2])
                tmpy = tmpy.view(-1, self.num_views, tmpy.shape[1], tmpy.shape[2])
                tmp_weights = torch.softmax(tmp_weights, dim=1)
                tmpy = (tmp_weights * tmpy).sum(dim=1)
            
        if self.last_op:
            y = self.last_op(y)
        return y    
    
class AttenSurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=6, no_residual=True, last_op=None):
        super(AttenSurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        b,c,n = feature.shape
        
        feature = feature.reshape(-1, self.num_views, c, n)
        feature = feature.permute(0, 3, 1, 2).reshape(-1, self.num_views, c)
        att = torch.matmul(feature, feature.transpose(1, 2))
        att = torch.softmax(att, dim=2)
        feature = torch.matmul(att, feature).reshape(-1, n, self.num_views, c).permute(0, 2, 3, 1).reshape(-1, c, n)
            
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
                
            if self.num_views > 1 and i == 1:
                y = (y.view(-1, self.num_views, y.shape[1], y.shape[2])).mean(1)
                tmpy = (feature.view(-1, self.num_views, feature.shape[1], feature.shape[2])).mean(1)
            
            
            if(i==0):
                c = y.shape[1]
                # print(y.hspae)
                y = y.reshape(-1, self.num_views, c, n)
                y = y.permute(0, 3, 1, 2).reshape(-1, self.num_views, c)
                att = torch.matmul(y, y.transpose(1, 2))
                att = torch.softmax(att, dim=2)
                y = torch.matmul(att, y).reshape(-1, n, self.num_views, c).permute(0, 2, 3, 1).reshape(-1, c, n)
            
            
        if self.last_op:
            y = self.last_op(y)
        return y 
    
    
    
class SpatitalSurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=6, no_residual=True, last_op=None):
        super(SpatitalSurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 1 == l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                elif l>1:
                     self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0] + 72,
                            filter_channels[l + 1],
                            1))                   
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))
                
                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature, spatital_feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                if(i==1):
                    y = torch.cat([y, tmpy], 1)
                elif(i>1):
                     y = torch.cat([y, tmpy, spatital_feature], 1)
                y = self._modules['conv' + str(i)](y)
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
                
            if self.num_views > 1 and i == 1:
                y = (y.view(-1, self.num_views, y.shape[1], y.shape[2])).mean(1)
                tmpy = (feature.view(-1, self.num_views, feature.shape[1], feature.shape[2])).mean(1)
            
        if self.last_op:
            y = self.last_op(y)
        return y



class SpatitalSurfaceClassifierV1(nn.Module):
    def __init__(self, filter_channels, num_views=6, no_residual=True, last_op=None):
        super(SpatitalSurfaceClassifierV1, self).__init__()
        self.filters = []
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.last_op:
                y = self.last_op(y)  
        return y


