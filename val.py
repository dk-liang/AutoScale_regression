from __future__ import division

import math
import pickle
import warnings
from functools import partial

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fpn_onnx import AutoScale
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms

import dataset
from find_couter import findmaxcontours
from image import *
from rate_model import RATEnet

warnings.filterwarnings('ignore')
from config import args
import  os
torch.cuda.manual_seed(args.seed)

def main():

    if args.test_dataset == 'ShanghaiA':
        test_file = './ShanghaiA_test.npy'
    elif args.test_dataset == 'ShanghaiB':
        test_file = './ShanghaiB_test.npy'
    elif args.test_dataset =='UCF_QNRF':
        test_file = './Qnrf_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model = AutoScale().cuda()
    model = nn.DataParallel(model, device_ids=[0])

    rate_model = RATEnet()
    rate_model = nn.DataParallel(rate_model, device_ids=[0]).cuda()

    pickle.load = partial(pickle.load, encoding="iso-8859-1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="iso-8859-1")

    if args.pre: 
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            # checkpoint = torch.load(args.pre, map_location=lambda storage, loc: storage, pickle_module=pickle)
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['pre_state_dict'])
            # rate_model.load_state_dict(checkpoint['rate_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    torch.save({
            'state_dict': model.state_dict()
        }, "./model/UCF_QNRF/model_best.pth")
    torch.save(model.module, "./model/UCF_QNRF/model_test_all.pt")

    validate(val_list, model, rate_model, args)


def target_transform(gt_point, rate):
    point_map = gt_point.cpu().numpy()
    pts = np.array(list(zip(np.nonzero(point_map)[2], np.nonzero(point_map)[1])))
    pt2d = np.zeros((int(rate * point_map.shape[1]) + 1, int(rate * point_map.shape[2]) + 1), dtype=np.float32)

    for i, pt in enumerate(pts):
        pt2d[int(rate * pt[1]), int(rate * pt[0])] = 1.0

    return pt2d


def gt_transform(pt2d, cropsize, rate):
    [x, y, w, h] = cropsize
    pt2d = pt2d[int(y * rate):int(rate * (y + h)), int(x * rate):int(rate * (x + w))]
    density = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    pts = np.array(list(zip(np.nonzero(pt2d)[1], np.nonzero(pt2d)[0])))
    orig = np.zeros((int(pt2d.shape[0]), int(pt2d.shape[1])), dtype=np.float32)
    for i, pt in enumerate(pts):
        orig[int(pt[1]), int(pt[0])] = 1.0

    density += scipy.ndimage.filters.gaussian_filter(orig, 4, mode='constant')
    # print(np.sum(density))
    return density

def validate(Pre_data, model, rate_model, args):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args.task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse = 0
    original_mae = 0
    visi = []
    density_threshold = 0.0005
    rate = torch.ones(1)

    for i, (img,target,kpoint,fname,sigma_map) in enumerate(test_loader):

        img = img.cuda()
        target = target.cuda()
        # print(img.shape,target.shape)
        # d2, d3, d4, d5, d6, fs = model(img, target, refine_flag=True)
        d6 = model(img)

        density_map = d6.data.cpu().numpy()
        original_count = density_map.sum()
        original_density = d6
        [x, y, w, h] = findmaxcontours(density_map, density_threshold, fname)

        # rate_feature = F.adaptive_avg_pool2d(fs[:, :, y:(y + h), x:(x + w)], (14, 14))
        # rate = rate_model(rate_feature).clamp_(0.5, 9)
        # rate = torch.sqrt(rate)

        if (float(w * h) / (img.size(2) * img.size(3))) > args.area_threshold:

            img_pros = img[:, :, y:(y + h), x:(x + w)]

            img_transed = F.upsample_bilinear(img_pros, scale_factor=rate.item())

            pt2d = target_transform(kpoint, rate)
            target_choose = gt_transform(pt2d, [x, y, w, h], rate.item())

            target_choose = torch.from_numpy(target_choose).type(torch.FloatTensor).unsqueeze(0)
            dd2, dd3, dd4, dd5, dd6 = model(img_transed, target_choose, refine_flag=False)

            # dd6[dd6<0]=0
            temp = dd6.data.cpu().numpy().sum()
            original_density[:, :, y:(y + h), x:(x + w)] = 0
            count = original_density.data.cpu().numpy().sum() + temp

        else:
            count = d6.data.cpu().numpy().sum()

        mae += abs(count - target.data.cpu().numpy().sum())
        mse += abs(count - target.data.cpu().numpy().sum()) * abs(count - target.data.cpu().numpy().sum())
        original_mae += abs(original_count - target.data.cpu().numpy().sum())

        if i % args.print_freq == 0:
            print(fname[0], 'rate {rate:.3f}'.format(rate=rate.item()), 'gt', int(target.data.cpu().numpy().sum()),
                  "pred", int(count), "original:", int(original_count))

    mae = mae / len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
    original_mae = original_mae / len(test_loader)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}\n'.format(mse=mse),'* ORI_MAE {ori_mae:.3f}\n'.format(ori_mae=original_mae))

    return mae, original_mae, visi

if __name__ == '__main__':
    main()
