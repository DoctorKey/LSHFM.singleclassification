import torch
import torch.nn as nn
import torch.nn.functional as F


class L2(nn.Module):
    def __init__(self, l2_norm=False):
        super(L2, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.l2_norm = l2_norm

    def forward(self, f_s, f_t):
        if self.l2_norm:
            f_t = F.normalize(f_t, p=2, dim=-1)
            f_s = F.normalize(f_s, p=2, dim=-1)
        return self.loss(f_s, f_t)
