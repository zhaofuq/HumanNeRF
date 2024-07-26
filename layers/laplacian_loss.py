import torch
import torch.nn as nn


class Laplacian_op(nn.Module):
    def __init__(self):
        super(Laplacian_op, self).__init__()
        # kernel = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        kernel = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        kernel = kernel.reshape((1, 1, 3, 3))
        self.sobel_op = nn.Conv2d(1, 1, 3, bias=False)
        self.sobel_op.weight.data = kernel

    def forward(self, x):
        n_channels = x.size(1)
        res = []
        for i in range(n_channels):
            res.append(self.sobel_op(x[:, i:i + 1, :, :]))
        return torch.cat(res, dim=1)


class Laplacian_loss(nn.Module):
    def __init__(self):
        super(Laplacian_loss, self).__init__()
        self.lop = Laplacian_op()
        self.lop.cuda()
        self.lop.eval()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        x, target = self.lop(x), self.lop(target)
        return self.loss(x, target)
