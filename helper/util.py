from __future__ import print_function

import socket
import random
import torch
import numpy as np


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_model(model, path, LOG, DataParallel=False):
    LOG.info("=> load from imagenet.py results")
    # state_dict keys begin with module.
    state_dict = torch.load(path)['state_dict']
    if not DataParallel:
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}       
    ret = model.load_state_dict(state_dict, strict=False)
    LOG.info("=> load result: {}".format(ret[0]))
    return model

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Tensorboard_logger(object):
    """docstring for Tensorboard_logger"""
    def __init__(self, save_dir):
        super(Tensorboard_logger, self).__init__()
        self.save_dir = save_dir
        hostname = socket.gethostname()
        if 'GPU2' in hostname or 'gpu1' in hostname:
            self.logger = None
        else:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(save_dir)

    def add_scalar(self, name, value, step):
        if self.logger:
            self.logger.add_scalar(name, value, step)

if __name__ == '__main__':

    pass
