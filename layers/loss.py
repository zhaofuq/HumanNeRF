import torch
import torch.nn as nn

def make_loss(cfg):
    return nn.MSELoss()

def make_weighted_loss(cfg):
    return WeightedLoss(reduction='mean')


def make_perceptual_loss(cfg):
    return Perceptual_loss()


def make_laplacian_loss(cfg):
    return Laplacian_loss()


def lab_distance(rgb_1, rgb_2):
    rgb_1, rgb_2 = rgb_1 * 255, rgb_2 * 255
    r_1, g_1, b_1 = rgb_1[:, 0], rgb_1[:, 1], rgb_1[:, 2]
    r_2, g_2, b_2 = rgb_2[:, 0], rgb_2[:, 1], rgb_2[:, 2]
    rmean = (r_1 + r_2) / 2
    r = r_1 - r_2
    g = g_1 - g_2
    b = b_1 - b_2
    d = torch.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (b ** 2))
    return d / 256


class WeightedLoss(nn.Module):
    def __init__(self, reduction='mean', loss_type='L2'):
        super(WeightedLoss, self).__init__()
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'L2':
            self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, src, tgt, weight=None):
        if weight is None:
            weight = torch.ones_like(src, device=src.device)
        return self.loss_fn(weight * src, weight * tgt), self.loss_fn(src, tgt)
