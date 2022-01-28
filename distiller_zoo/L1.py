import torch
import torch.nn as nn


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, f_s, f_t):
        #f_t = f_t * 2
        return self.loss(f_s, f_t)
