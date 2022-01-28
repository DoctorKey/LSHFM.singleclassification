
import os
import argparse
import socket
import time
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, set_random_seed, Tensorboard_logger

from distiller_zoo import LSH, DistillKL, L1, L2, LSH_S

from helper.loops import train_distill as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=120, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('--gpus', type=str, default='0', help='choose gpu')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='lshl2_s', choices=['l1', 'l2', 'lsh', 'lshl2', 'lshl2_s'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=6, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    parser.add_argument('--hash_num', default=None, type=int, help='feature dimension (default: 32D)')
    parser.add_argument('--std', default=None, type=float, help='the std of LSH weight, default is the std of teacher fc weight')
    parser.add_argument('--bias', type=str, default='median', choices=['0', 'mean', 'median'])
    parser.add_argument('--LSH_loss', type=str, default='BCE', choices=['BCE', 'L1', 'L2'])

    parser.add_argument('--force_2FC', dest='force_2FC', action='store_true',
                    help='use 2FC in student')

    parser.add_argument('--seed', type=int, default=0, help='the seed of randomness')

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/train_student'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    model_t = opt.model_t
    model_s = opt.model_s
    if opt.force_2FC:
        model_s = model_s + '(2FC)'

    opt.model_name = '{}_S:{}_T:{}_r:{}_b:{}_N:{}_std:{}_bias:{}_LSHloss:{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
        opt.distill, model_s, model_t, opt.gamma, opt.beta, opt.hash_num, opt.std, opt.bias, opt.LSH_loss, date=datetime.now())

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    set_random_seed(opt.seed)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls, opt):
    LOG = logging.getLogger('main')
    LOG.info('==> loading teacher model from {}'.format(model_path))
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path)['model'])
    LOG.info('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = Tensorboard_logger(opt.tb_folder)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    LOG = logging.getLogger('main')
    FileHandler = logging.FileHandler(os.path.join(opt.save_folder, 'log.txt'))
    LOG.addHandler(FileHandler)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    else:
        raise ValueError(opt.dataset)


    data = torch.randn(2, 3, 32, 32)

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt)
    model_t.eval()
    feat_t, _ = model_t(data, is_feat=True)
    #t_fc_weight, t_fc_bias = model_t.get_classifier_weight()
    opt.t_dim = feat_t[-1].shape[1]
    weight, bias = model_t.get_classifier_weight()
    LOG.info('=> teacher classifier weight std: {}'.format(weight.std()))
    if opt.std is None:
        opt.std = weight.std()
    if opt.hash_num is None:
        opt.hash_num = 32 * opt.t_dim

    model_s = model_dict[opt.model_s](num_classes=n_cls, student_dim=opt.t_dim, force_2FC=opt.force_2FC)

    
    if opt.path_s is not None:
        LOG.info("=> loading student from {}".format(opt.path_s))
        state_dict = torch.load(opt.path_s)['model']
        if model_s.feat_fc:
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            state_dict.pop('classifier.weight', None)
            state_dict.pop('classifier.bias', None)
            state_dict.pop('linear.weight', None)
            state_dict.pop('linear.bias', None)
        ret = model_s.load_state_dict(state_dict, strict=False)
        LOG.info("=> load student result: {}".format(ret))


    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)


    LOG.info('=> creating {} for knowledge distillation'.format(opt.distill))
    if opt.distill == 'l1':
        criterion_kd = L1()
    elif opt.distill == 'l2':
        criterion_kd = L2()
    elif opt.distill == 'lsh':
        LOG.info('=> LSH: D:{} N:{} std:{} LSH_loss:{}'.format(opt.t_dim, opt.hash_num, opt.std, opt.LSH_loss))
        criterion_kd = LSH(opt.t_dim, opt.hash_num, opt.std, with_l2=False, LSH_loss=opt.LSH_loss)
    elif opt.distill == 'lshl2':
        LOG.info('=> LSHl2: D:{} N:{} std:{} LSH_loss:{}'.format(opt.t_dim, opt.hash_num, opt.std, opt.LSH_loss))
        criterion_kd = LSH(opt.t_dim, opt.hash_num, opt.std, with_l2=True, LSH_loss=opt.LSH_loss)
    elif opt.distill == 'lshl2_s':
        LOG.info('=> LSHl2_S: D:{} N:{} std:{}'.format(opt.t_dim, opt.hash_num, opt.std))
        criterion_kd = LSH_S(opt.t_dim, opt.hash_num, opt.std, with_l2=True)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(model_s.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    if 'lsh' in opt.distill:
        if opt.bias == '0':
            LOG.info('=> init LSH bias by 0')
        elif opt.bias == 'median':
            LOG.info('=> init LSH bias by median')
            criterion_kd.init_bias(model_t, train_loader, opt.print_freq, use_median=True)
        elif opt.bias == 'mean':
            LOG.info('=> init LSH bias by mean')
            criterion_kd.init_bias(model_t, train_loader, opt.print_freq, use_median=False)
        else:
            raise NotImplementedError(opt.bias)


    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    LOG.info('teacher accuracy: {}'.format(teacher_acc))

    avg_state_dict = None

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        LOG.info("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        LOG.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('test_loss', test_loss, epoch)
        logger.add_scalar('test_acc_top5', tect_acc_top5, epoch)

        if epoch > opt.epochs - 10:
            if avg_state_dict is not None:
                state_dict = model_s.state_dict()
                for key in avg_state_dict.keys():
                    avg_state_dict[key] = avg_state_dict[key] + state_dict[key]
            else:
                avg_state_dict = model_s.state_dict()

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'criterion_kd': criterion_kd.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            LOG.info('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'criterion_kd': criterion_kd.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            LOG.info('==> Saved to {}'.format(save_file))

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    LOG.info('best accuracy:{}'.format(best_acc))

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'criterion_kd': criterion_kd.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)
    LOG.info('==> Saved to {}'.format(save_file))

    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / 10.

    model_s.load_state_dict(avg_state_dict)
    LOG.info("=> Test avg last 10")
    test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'criterion_kd': criterion_kd.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_avg_last_10.pth'.format(opt.model_s))
    torch.save(state, save_file)
    LOG.info('==> Saved to {}'.format(save_file))


if __name__ == '__main__':
    main()
