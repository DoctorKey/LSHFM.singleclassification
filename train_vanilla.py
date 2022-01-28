from __future__ import print_function

import os
import argparse
import socket
import time
import logging
from datetime import datetime

#import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter, set_random_seed, Tensorboard_logger
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=400, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('--gpus', type=str, default='0', help='choose gpu')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--ft_fc', default=False, action='store_true',
                        help='only finetune fc')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet18', 'ResNet50'])
    parser.add_argument('--embed-fc', type=int, default=0, help='dim of embedding FC')
    parser.add_argument('--path', type=str, default=None, help='model snapshot')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/train_vanilla'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.path is not None:
        opt.model_name = '{}_finetune_lr_{}_decay_{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
            opt.model, opt.learning_rate, opt.weight_decay, date=datetime.now())
    else:
        opt.model_name = '{}_lr_{}_decay_{}_{date:%Y-%m-%d_%H:%M:%S}'.format(
            opt.model, opt.learning_rate, opt.weight_decay, date=datetime.now())

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    set_random_seed(0)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    logger = Tensorboard_logger(opt.tb_folder)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    LOG = logging.getLogger('main')
    FileHandler = logging.FileHandler(os.path.join(opt.save_folder, 'log.txt'))
    LOG.addHandler(FileHandler)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    LOG.info("=> finetune FC: {}".format(opt.ft_fc))

    # model
    model = model_dict[opt.model](num_classes=n_cls, student_dim=opt.embed_fc)
    if opt.path is not None:
        LOG.info("=> loading {}".format(opt.path))
        state_dict = torch.load(opt.path)['model']
        if opt.ft_fc:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        LOG.info("=> load result: {}".format(ret))
        

    # optimizer
    if opt.ft_fc:
        opt_param = model.fc.parameters()
    else:
        opt_param = model.parameters()
    optimizer = optim.SGD(opt_param,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        LOG.info("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        LOG.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('test_acc_top5', test_acc_top5, epoch)
        logger.add_scalar('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            LOG.info('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            LOG.info('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    LOG.info('best accuracy:{}'.format(best_acc))

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
