import argparse
import os
import random
import shutil
import time
import warnings
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from distiller_zoo import LSH, DistillKL, LSH_S
from helper.util import load_model, set_random_seed


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet18',
                        choices=['ResNet18', 'resnet50', 'mobilenetv2'])
parser.add_argument('--teacher-arch', metavar='ARCH', default='ResNet34',
                        choices=['ResNet34', 'ResNet101'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=str, default='schedule',
                    help='mode for learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma for LR decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', type=str, default='0', help='choose gpu')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


parser.add_argument('--distill', type=str, default='lshl2_s', choices=['lshl2', 'lshl2_s', 'lsh_s'])
parser.add_argument('--force_2FC', dest='force_2FC', action='store_true',
                    help='use 2FC in student')

parser.add_argument('--feat-dim', default=2048, type=int, help='feature dimension')
parser.add_argument('-b', '--beta', type=float, default=5, help='weight balance for lsh losses')
parser.add_argument('--lsh_std', type=float, default=None, help='std for lsh')
# KL distillation
parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--alpha', type=float, default=0, help='weight balance for KD')

parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')

best_acc1 = 0

teacher_weight = {
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

teacher_feature_dim = {
    'ResNet34': 512,
    'ResNet101': 2048,
}

teacher_fc_std = {
    'ResNet34': 0.06,
    'ResNet101': 0.03,
}

def main():
    args = parser.parse_args()

    set_random_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.lr_decay == 'schedule':
        decay_epoch = 30
        args.schedule = [decay_epoch * (i + 1) for i in range(args.epochs // decay_epoch)]

    if args.lsh_std is None:
        args.lsh_std = teacher_fc_std.get(args.teacher_arch)
    save_folder = '{}_{}_b:{}_std:{}_dim:{}_a:{}_T:{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
        args.arch, args.distill, args.beta, args.lsh_std, args.feat_dim, args.alpha, args.kd_T, date=datetime.now())


    tb_folder = os.path.join('./save/student_tensorboards', 'imagenet_lsh_' + save_folder)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)
    tb_logger = SummaryWriter(tb_folder)

    log_folder = os.path.join('./save/imagenet_lsh', save_folder)
    args.log_folder = log_folder
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    LOG = logging.getLogger('main')
    FileHandler = logging.FileHandler(os.path.join(log_folder, 'log.txt'))
    LOG.addHandler(FileHandler)

    LOG.info("=> inited log")
    LOG.info("=> use gpus: {}".format(args.gpus))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, tb_logger, LOG))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, tb_logger, LOG)


def main_worker(gpu, ngpus_per_node, args, tb_logger, LOG):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        LOG.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    f_t_dim = teacher_feature_dim.get(args.teacher_arch)
    LOG.info("=> creating student model '{}'".format(args.arch))
    model_s = model_dict[args.arch](student_dim=f_t_dim, force_2FC=args.force_2FC)
    if args.path_s:
        LOG.info("=> loading student model from '{}'".format(args.path_s))
        model_s = load_model(model_s, args.path_s, LOG, DataParallel=False)

    LOG.info("=> creating teacher model '{}'".format(args.teacher_arch))
    model_t = model_dict[args.teacher_arch]()
    for param in model_t.parameters():
        param.detach_()
    if args.path_t:
        LOG.info("=> loading teacher model from '{}'".format(args.path_t))
        model_t = load_model(model_t, args.path_t, LOG, DataParallel=False)
    else:
        url = teacher_weight.get(args.teacher_arch)
        LOG.info("=> loading teacher model from '{}'".format(url))
        state_dict = torch.hub.load_state_dict_from_url(url)
        ret = model_t.load_state_dict(state_dict)
        LOG.info("=> load result: {}".format(ret))

    model_s = torch.nn.DataParallel(model_s).cuda()
    model_t = torch.nn.DataParallel(model_t).cuda()

    LOG.info("=> creating {}".format(args.distill))
    if args.distill == 'lshl2':
        criterion_kd = LSH(f_t_dim, args.feat_dim, args.lsh_std).cuda(args.gpu)
    elif args.distill == 'lshl2_s':
        criterion_kd = LSH_S(f_t_dim, args.feat_dim, args.lsh_std).cuda(args.gpu)
    elif args.distill == 'lsh_s':
        criterion_kd = LSH_S(f_t_dim, args.feat_dim, args.lsh_std, with_l2=False).cuda(args.gpu)
    else:
        raise NotImplementedError(args.distill)
    #LOG.info('=> init LSH bias by mean')
    #criterion_kd.init_bias(model_t, train_loader, args.print_freq, use_median=False)
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)
    

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    optimizer = torch.optim.SGD(model_s.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            LOG.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model_s.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            LOG.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            LOG.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model_s, criterion, args)
        return

    avg_state_dict = None
    LOG.info("=> begin training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        time1 = time.time()
        train_acc, train_loss = train(train_loader, model_s, model_t, criterion_list, optimizer, epoch, args)
        LOG.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - time1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            tb_logger.add_scalar('train_acc', train_acc, epoch)
            tb_logger.add_scalar('train_loss', train_loss, epoch)

        # evaluate on validation set
        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion, args)
        LOG.info(' * TEST {} Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
              .format(epoch, top1=test_acc, top5=tect_acc_top5))

        if epoch+1 > args.epochs - 10:
            if avg_state_dict is not None:
                state_dict = model_s.state_dict()
                for key in avg_state_dict.keys():
                    avg_state_dict[key] = avg_state_dict[key] + state_dict[key]
            else:
                avg_state_dict = model_s.state_dict()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            tb_logger.add_scalar('test_acc', test_acc, epoch)
            tb_logger.add_scalar('test_loss', test_loss, epoch)
            tb_logger.add_scalar('test_acc_top5', tect_acc_top5, epoch)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc1
        best_acc1 = max(test_acc, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_file = os.path.join(args.log_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_s.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=save_file)

    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / 10.

    model_s.load_state_dict(avg_state_dict)
    test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion, args)
    LOG.info(' * Avg 10 Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
          .format(top1=test_acc, top5=tect_acc_top5))
    save_file = os.path.join(args.log_folder, 'ckpt_epoch_avg10.pth')
    save_checkpoint({
        'epoch': 'avg10',
        'arch': args.arch,
        'state_dict': model_s.state_dict(),
        'best_acc1': best_acc1,
    }, False, filename=save_file)

    save_file = os.path.join(args.log_folder, 'ckpt_epoch_avg10.pth')
    save_checkpoint({
        'epoch': 'avg10',
        'arch': args.arch,
        'state_dict': model_s.state_dict(),
        'best_acc1': best_acc1,
    }, is_best, filename=save_file)


def train(train_loader, model_s, model_t, criterion_list, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    losses = AverageMeter('Loss', ':6.3f')
    cls_losses = AverageMeter('ClsLoss', ':5.3f')
    div_losses = AverageMeter('DivLoss', ':4.2f')
    lsh_losses = AverageMeter('LSHLoss', ':5.3f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cls_losses, div_losses, lsh_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    criterion = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    # switch to train mode
    model_s.train()
    model_t.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        adjust_learning_rate(optimizer, epoch, args, i, len(train_loader))

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        if args.distill == 'lshl2':
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1])
        elif args.distill in ['lshl2_s', 'lsh_s']:
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1], logit_t, target)

        loss = loss_cls + args.alpha * loss_div + args.beta * loss_kd

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        cls_losses.update(loss_cls.item(), images.size(0))
        div_losses.update(loss_div.item(), images.size(0))
        lsh_losses.update(loss_kd.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        #LOG.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        dirname = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log = logging.getLogger('main')

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

'''
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
from math import cos, pi
def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_iter = 0
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
