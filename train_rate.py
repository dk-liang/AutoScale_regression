import sys
import os

import warnings

from rate_model import RATEnet
# from crop_half import crop_half
from utils import save_checkpoint
from centerloss import CenterLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import argparse
import json
import cv2
import rate_dataloader
import time
import math
from PIL import Image
from rate_img import *

parser = argparse.ArgumentParser(description='PyTorch CSRNet')


# parser.add_argument('train_json', metavar='TRAIN',
#                     help='path to train json')
# parser.add_argument('test_json', metavar='TEST',
#                     help='path to test json')
#
# parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
#                     help='path to the pretrained model')
#
# parser.add_argument('gpu',metavar='GPU', type=str,
#                     help='GPU id to use.')
#
# parser.add_argument('task',metavar='TASK', type=str,
#                     help='task id to use.')
# parser.add_argument('density_value',metavar='DENSITY_VALUE',type=float,help='density value threthod')

def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-3
    args.lr = 1e-2
    args.rate_lr = 1e-2
    args.center_lr = 1e-1
    args.batch_size = 300
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 5000
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    args.density_value = 3

    args.task = "save_file_ratemodel"

    with open('./ShanghaiAfeature.npy', 'r') as outfile:
        train_list = np.load(outfile).tolist()
    with open('./ShanghaiAfeature.npy', 'r') as outfile:
        val_list = np.load(outfile).tolist()
    print (len(train_list),train_list[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    density_value = args.density_value

    torch.cuda.manual_seed(args.seed)

    model = RATEnet()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = CenterLoss(1, 1).cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': criterion.parameters(), 'lr': args.center_lr},
            {'params': model.module.parameters(), 'lr': args.rate_lr}
        ], lr=1e-4)

    args.pre = './fpn_scale/model_best1.pth.tar'
    args.pre = None
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # print(checkpoint['state_dict'].keys())
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    # Pre_data_train = pre_data(train_list,train=True)
    # Pre_data_val = pre_data(val_list,train=False)
    print(len(train_list))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        flag = train(train_list, model, criterion, optimizer, epoch, args.task, density_value)
        # prec1,visi= validate(val_list, model, criterion,args.task,density_value)
        # if flag == 1 :
        #     break
        visi = []
        is_best = True
        ori_isbest = False
        # is_best = prec1 < best_prec1
        # best_prec1 = min(prec1, best_prec1)
        # print(' * best MAE {mae:.9f} '
        #       .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, ori_isbest, args.task)


def pre_data(train_list, train):
    print "Pre_load dataset ......"
    data_keys = {}
    if train:
        train_list = train_list
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, target = load_data(Img_path, train)
        blob = {}
        blob['img'] = img
        blob['gt'] = target
        blob['fname'] = fname
        data_keys[j] = blob
    return data_keys


def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]

    d1 = d[:, :, abs(int(math.floor((d_h - g_h) / 2.0))):abs(int(math.floor((d_h - g_h) / 2.0))) + g_h,
         abs(int(math.floor((d_w - g_w) / 2.0))):abs(int(math.floor((d_w - g_w) / 2.0))) + g_w]
    return d1


def choose_crop(output, target):
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] > target.size()[2]) | (output.size()[3] > target.size()[3]):
        output = crop(output, target)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    if (output.size()[2] < target.size()[2]) | (output.size()[3] < target.size()[3]):
        target = crop(target, output)
    return output, target


def train(Pre_data, model, criterion, optimizer, epoch, task_id, density_value):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        rate_dataloader.listDataset(Pre_data, task_id,
                                    shuffle=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),
                                    ]),
                                    train=True,
                                    seen=model.module.seen,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers),
        batch_size=args.batch_size)
    # print('epoch %d, processed %d samples, lr %.13f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    with open('./ShanghaiAfeature.npy', 'r') as outfile:
        train_list = np.load(outfile).tolist()

    for i, (img,  fname) in enumerate(train_loader):
        mean_d = []
        data_time.update(time.time() - end)
        # print(img.shape)
        # print(img.size(),target.size(),density.size())
        # target = target.squeeze(1)
        # img,target = crop_half(img,target,density_value,train=True)
        # print(img.size(),target.size())
        img = img.squeeze(1).cuda()
        img = Variable(img)
        # print(img.size())
        scale_factor = model(img)

        sigma_count_list = []
        gt_num_list = []

        for j in range(len(scale_factor)):

            path = train_list[j]

            # print(path)
            gt_file = h5py.File(path)
            # print(path)
            sigma = np.asarray(gt_file['sigma'])
            gt_num = np.asarray(gt_file['gt_num'])

            sigma = torch.from_numpy(sigma).cuda()
            sigma = Variable(sigma)
            sigma_count = torch.sum(sigma).type(torch.FloatTensor).cuda()
            sigma_count_list.append(sigma_count)

            gt_num = torch.from_numpy(gt_num).type(torch.FloatTensor).cuda()
            if gt_num == 0:
                gt_num = 1
            gt_num_list.append(gt_num)
            mean_d.append(sigma_count / gt_num * scale_factor[j])

        center_sample = torch.stack(mean_d, 0).view(len(mean_d), -1)
        center_label = torch.zeros(len(mean_d)).type(torch.FloatTensor).cuda()

        centerloss = criterion(center_sample, center_label)
        loss = centerloss
        average_distance = np.sum(sigma_count_list)/np.sum(gt_num_list)
        # args.lr = args.original_lr*density_array.size(1)

        # print(criterion.centers)

        # print(output.size(),density_map.size())
        # target = target.type(torch.FloatTensor).cuda()
        # target = Variable(target)

        # # print(output.size(),density_map.size(),target.size())
        # loss =criterion(d2,target) +criterion(d3, target)+criterion(d4, target)+criterion(d5, target)+criterion(d6, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if loss<0.0000020:
        #     return 1
        batch_time.update(time.time() - end)
        end = time.time()

        # print(epoch%args.print_freq)
        if epoch % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            print( "max:", torch.max(scale_factor).item(), "min:",
                  torch.min(scale_factor).item(), "ave:", torch.mean(scale_factor).item(),
                  "rate5", scale_factor[5], "rate123", scale_factor[123],"distance",average_distance)


def validate(Pre_data, model, criterion, task_id, density_value):
    print ('begin test')
    # Pre_data = pre_data(val_list,train=False)
    test_loader = torch.utils.data.DataLoader(
        rate_dataloader.listDataset(Pre_data, task_id,
                                    shuffle=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225]),
                                    ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    var = 0
    visi = []
    for i, (img, target, fname) in enumerate(test_loader):
        # img,target = crop_half(img,target,density_value,train=False)

        target = target.squeeze(1)
        # img,target = crop_half(img,target,density_value,train=True)
        # print(img.size(),target.size())
        img = img.squeeze(1).cuda()
        img = Variable(img)
        scale_factor = model(img)
        # print(scale_factor.size()
        #       scale_factor.clamp_(0.64,10)
        target_density = F.adaptive_avg_pool2d(target, (2, 2)).type(torch.FloatTensor).cuda()
        density = (target_density / scale_factor)

        density_array = density[target_density > 0.002]
        print(torch.max(scale_factor), torch.min(scale_factor), torch.mean(scale_factor), scale_factor[0, 0, 1, 1])

        visi.append([img.data.cpu().numpy(), target.data.cpu().numpy(), fname])
        var += torch.var(density_array)

    var = var / len(test_loader)
    print(' * var {var:.9f} '
          .format(var=var))

    return var, visi


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


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


if __name__ == '__main__':
    main()        
