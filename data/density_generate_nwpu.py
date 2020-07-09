
# coding: utf-8

# In[1]:


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

import  cv2


'''please set your dataset path'''
NWPU_Crowd_path = '/home/dkliang/projects/synchronous/NWPU_regression/images_1024/'

path_sets = [NWPU_Crowd_path]

if not os.path.exists(NWPU_Crowd_path.replace('images','gt_density_map')):
    os.makedirs(NWPU_Crowd_path.replace('images','gt_density_map'))
    


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:


    img = cv2.imread(img_path)

    k = np.zeros((img.shape[0] ,img.shape[1] ))
    mat_path = img_path.replace('images', 'gt_npydata').replace('jpg','npy')

    with open(mat_path, 'rb') as outfile:
        gt = np.load(outfile).tolist()

    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            # print(gt[i][1],gt[i][0])
            k[int(gt[i][1]), int(gt[i][0])] = 1

    kpoint = k.copy()
    k = gaussian_filter(k, 6)

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree

    if int(kpoint.sum()) >1:
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

    with h5py.File(img_path.replace('images', 'gt_density_map').replace('jpg','h5'), 'w') as hf:
            hf['density_map'] = k
            hf['kpoint'] = kpoint
            hf['sigma_map'] = sigma_map
    density_map = k
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)


    print(img_path)
    gt_show = img_path.replace('images','gt_show_density')
    cv2.imwrite(gt_show,density_map)





print ("end")