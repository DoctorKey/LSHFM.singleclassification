from __future__ import print_function, division

import sys
import time
import logging
import torch
import torch.nn.functional as F

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    LOG = logging.getLogger('main')
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if len(data) == 4:
            input, fv, target, index = data
        elif len(data) == 3:
            input, target, index = data
        elif len(data) == 2:
            input, target = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if len(data) == 4:
                fv = fv.float().cuda()

        # ===================forward=====================
        if len(data) == 3:
            output = model(input, index)
        elif len(data) == 2:
            output = model(input)
        elif len(data) == 4:
            output = model(fv)

        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if idx % opt.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    LOG.info(' * Train Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    LOG = logging.getLogger('main')
    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    #criterion_kd.reinit()  

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # ===================forward=====================
        feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=False)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        f_s = feat_s[-1]
        f_t = feat_t[-1]
        if opt.distill == 'lshl2_s':
            loss_kd = criterion_kd(f_s, f_t, logit_t, target)
        else:
            loss_kd = criterion_kd(f_s, f_t)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        cls_losses.update(loss_cls.item(), input.size(0))
        kd_losses.update(loss_kd.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Cls {cls_losses.val:.4f} ({cls_losses.avg:.4f})\t'
              'Kd {kd_losses.val:.4f} ({kd_losses.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
              .format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, cls_losses=cls_losses, 
                kd_losses=kd_losses, 
                top1=top1, top5=top5))

    LOG.info(' * Train Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} '
          .format(top1=top1, top5=top5))
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, opt):
    """validation"""
    LOG = logging.getLogger('main')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            if len(data) == 3:
                input, target, index = data
            elif len(data) == 2:
                input, target = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            if len(data) == 3:
                output = model(input, index)
            elif len(data) == 2:
                output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                LOG.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        LOG.info(' * Test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
