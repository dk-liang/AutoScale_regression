
# coding: utf-8

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import scipy
import json
from matplotlib import cm as CM

import torch
import  cv2


'''please set your dataset path'''
root = '/home/dkliang/projects/synchronous/jhu_crowd_v2.0'

#now generate the ShanghaiA's ground truth
train = root + '/train/images/'
val =  root + '/val/images/'
test = root +'/test/images/'

if not os.path.exists(train.replace('images','images_2048')):
    os.makedirs(train.replace('images','images_2048'))

if not os.path.exists(train.replace('images','gt_density_map_2048')):
    os.makedirs(train.replace('images','gt_density_map_2048'))

if not os.path.exists(train.replace('images','gt_show_density')):
    os.makedirs(train.replace('images','gt_show_density'))


if not os.path.exists(val.replace('images','images_2048')):
    os.makedirs(val.replace('images','images_2048'))

if not os.path.exists(val.replace('images','gt_density_map_2048')):
    os.makedirs(val.replace('images','gt_density_map_2048'))


if not os.path.exists(test.replace('images','images_2048')):
    os.makedirs(test.replace('images','images_2048'))

if not os.path.exists(test.replace('images','gt_density_map_2048')):
    os.makedirs(test.replace('images','gt_density_map_2048'))


    
    
path_sets = [test]
save_path = root + '/save_image/'

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()
count  = 0
for img_path in img_paths:
    # if img_path != '/data/weixu/jhu_crowd_v2.0/test/images/0597.jpg':
    #     continue
    count = count+1
    # if count<200:
    #     continue

    rate1 = 1
    rate2 = 1
    rate = 1

    img = cv2.imread(img_path)

    if img.shape[1]>=img.shape[0] and img.shape[1] >= 2048:
        rate1 = 2048.0 / img.shape[1]
    elif img.shape[0]>=img.shape[1] and img.shape[0] >= 2048:
        rate1 = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0,0), fx=rate1, fy=rate1)

    min_shape = 512.0
    if img.shape[1] <= img.shape[0] and img.shape[1] <= min_shape:
        rate2 = min_shape / img.shape[1]
    elif img.shape[0] <= img.shape[1] and img.shape[0] <= min_shape:
        rate2 = min_shape / img.shape[0]
    img = cv2.resize(img, (0,0), fx=rate2, fy=rate2)

    rate = rate1 * rate2


    k = np.zeros((img.shape[0] ,img.shape[1] ))
    gt_file = np.loadtxt(img_path.replace('images','gt').replace('jpg','txt'))

    fname = img_path.split('/')[6]
    print(count, img_path)
    try:
        gt_file.shape[1]
        y = gt_file[:, 0] * rate
        x = gt_file[:, 1] * rate

        for i in range(0, len(x)):
            if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                # print(gt[i][1],gt[i][0])
                k[int(x[i]), int(y[i])] = 1

    except Exception:

        try:

            y = gt_file[0] * rate
            x = gt_file[1] * rate

            for i in range(0, 1):
                if int(x) < img.shape[0] and int(y) < img.shape[1]:
                    # print(gt[i][1],gt[i][0])
                    k[int(x), int(y)] = 1
        except Exception:
            print("the image has zero person")


    kpoint = k.copy()
    kernel = 8
    k = gaussian_filter(k,  kernel)

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree


    if int(kpoint.sum()) > 1:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=2)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i][1]) / 2
            sigma_map[pt[1], pt[0]] = sigma
    elif int(kpoint.sum()) == 1:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=1)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i]) / 1
            sigma_map[pt[1], pt[0]] = sigma
    else:
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)

    with h5py.File(img_path.replace('images', 'gt_density_map_2048').replace('jpg','h5'), 'w') as hf:
            hf['density_map'] = k
            hf['kpoint'] = kpoint
            hf['kernel'] = kernel
            hf['sigma_map'] = sigma_map

    cv2.imwrite(img_path.replace('images','images_2048'), img)

    density_map = k
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)

    gt_show = img_path.replace('images', 'gt_show_density')
    cv2.imwrite(gt_show,density_map)
    #
    # print(img_path)



print ("end")