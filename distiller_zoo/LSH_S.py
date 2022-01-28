import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


class LSH_S(nn.Module):
    def __init__(self, input_dim, output_dim, std=1.0, with_l2=True):
        super(LSH_S, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std

        self.LSH_weight = nn.Linear(self.input_dim, self.output_dim, bias=True)
        if with_l2:
            self.mse_loss = torch.nn.MSELoss(reduction='none')
        else:
            self.mse_loss = None
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

        self._initialize()
        

    def _initialize(self):
        nn.init.normal_(self.LSH_weight.weight, mean=0.0, std=self.std)
        nn.init.constant_(self.LSH_weight.bias, 0)
        self.LSH_weight.weight.requires_grad_(False)
        self.LSH_weight.bias.requires_grad_(False)


    def init_bias(self, model_t, train_loader, print_freq=None, use_median=True):
        if use_median:
            print("=> Init LSH bias by median")
        else:
            print("=> Init LSH bias by mean")
        dataset_size = len(train_loader.dataset)
        if use_median:
            all_hash_value = torch.zeros(dataset_size, self.output_dim)
        else:
            mean = torch.zeros(self.output_dim)

        model_t.eval()

        for idx, data in enumerate(train_loader):
            input = data[0]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()

            # ============= forward ==============
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=False)
                feat_t = [f.detach() for f in feat_t]
                hash_t = self.LSH_weight(feat_t[-1])

            if use_median:
                index = data[-1]
                all_hash_value[index] = hash_t.cpu()
            else:
                mean += hash_t.sum(0).cpu() / dataset_size
            if print_freq is not None:
                if idx % print_freq == 0:
                    print("Init Bias: [{}/{}]".format(idx, len(train_loader)))

        if use_median:
            self.LSH_weight.bias.data[:] = - all_hash_value.median(0)[0]
        else:
            self.LSH_weight.bias.data[:] = - mean


    def forward(self, f_s, f_t, f_t_logit, label):
        hash_s = self.LSH_weight(f_s)
        hash_t = self.LSH_weight(f_t)
        hash_t = (hash_t > 0).float()
        loss = self.BCELoss(hash_s, hash_t)
        if self.mse_loss:
            l = self.mse_loss(f_s, f_t).mean(1) + loss.mean(1)
        else:
            l = loss.mean(1)
        predict = f_t_logit.argmax(1)
        pre_right = (predict == label).float().detach()
        l = (l * pre_right).sum() / pre_right.sum()
        return l

