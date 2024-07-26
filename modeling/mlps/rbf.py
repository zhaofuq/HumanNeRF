
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFunction(nn.Module):
    def __init__(self, filter_channels):
        super(RBFunction, self).__init__()

        self.filters = []
        filter_channels = filter_channels
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
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
        if i != len(self.filters) - 1:
            y = F.leaky_relu(y)

        return y