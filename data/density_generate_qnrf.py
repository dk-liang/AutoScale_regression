import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

'''please set your dataset path'''
root = './UCF-QNRF'


img_train_path = root + '/Train/'
gt_train_path = root + '/Train/'
img_test_path = root + '/Test/'
gt_test_path = root + '/Test/'

save_train_img_path = root + '/train_data/images/'
save_train_gt_path = root + '/train_data/gt_density_map/'
save_test_img_path = root + '/test_data/images/'
save_test_gt_path = root + '/test_data/gt_density_map/'

if not os.path.exists(save_train_img_path):
    os.makedirs(save_train_img_path)

if not os.path.exists(save_train_gt_path):
    os.makedirs(save_train_gt_path)

if not os.path.exists(save_train_img_path.replace('images', 'gt_show_density')):
    os.makedirs(save_train_img_path.replace('images', 'gt_show_density'))

if not os.path.exists(save_test_img_path):
    os.makedirs(save_test_img_path)

if not os.path.exists(save_test_gt_path):
    os.makedirs(save_test_gt_path)

if not os.path.exists(save_test_img_path.replace('images', 'gt_show_density')):
    os.makedirs(save_test_img_path.replace('images', 'gt_show_density'))

distance = 1
img_train = []
gt_train = []
img_test = []
gt_test = []

for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.split('.')[1] == 'mat':
        gt_train.append(file_name)

for file_name in os.listdir(img_test_path):
    if file_name.split('.')[1] == 'jpg':
        img_test.append(file_name)

for file_name in os.listdir(gt_test_path):
    if file_name.split('.')[1] == 'mat':
        gt_test.append(file_name)

img_train.sort()
gt_train.sort()
img_test.sort()
gt_test.sort()
# print(img_train)
# print(gt_train)
print(len(img_train),len(gt_train), len(img_test),len(gt_test))


min_x = 640
min_y = 480
x = []
y = []
count_min_x = 0
count_min_y = 0
start = time.time()

for k in range(len(img_train)):

    Img_data = cv2.imread(img_train_path + img_train[k])
    Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
    rate = 1
    flag = 0
    if Img_data.shape[1]>=Img_data.shape[0] and Img_data.shape[1] >= 1024:
        rate = 1024.0 / Img_data.shape[1]
        flag =1
    if Img_data.shape[0]>=Img_data.shape[1] and Img_data.shape[0] >= 1024:
        rate = 1024.0 / Img_data.shape[0]
        flag =1

    Img_data = cv2.resize(Img_data,(0,0),fx=rate,fy=rate)

    if k%100==0:
        print (img_train[k],Img_data.shape)

    # if Img_data.shape[0]<min_y:
    #     min_y = Img_data.shape[0]
    #     count_min_y = count_min_y+1
    #     print(min_x, min_y, img_train[k])
    #
    # if Img_data.shape[1]<min_x:
    #     min_x = Img_data.shape[1]
    #     count_min_x =count_min_x
    #     print(min_x,min_y, img_train[k])

    x.append(Img_data.shape[1])
    y.append(Img_data.shape[0])
    #print(img_train[k], min_y, min_x, rate, Img_data.shape)
    patch_x = Img_data.shape[1]/2
    patch_y = Img_data.shape[0]/2
    Gt_data = Gt_data['annPoints']

    Gt_data = Gt_data * rate

    density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    kpoint = density_map.copy()
    density_map = gaussian_filter(density_map, 6)

    new_img_path = (save_train_img_path + img_train[k])

    mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images', 'gt_show_density')
    h5_path = save_train_gt_path + img_train[k].replace('.jpg','.h5')

    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
    # pt2d = np.zeros(k.shape,dtype= np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1]) / 2
        sigma_map[pt[1], pt[0]] = sigma

    with h5py.File(h5_path, 'w') as hf:
        hf['density_map'] = density_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] = sigma_map

    cv2.imwrite(new_img_path, Img_data)
    density_map = density_map / np.max(density_map) * 255

    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)
    cv2.imwrite(gt_show_path, density_map)



for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])

    rate = 1
    flag = 0
    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >=1024:
        rate = 1024.0 / Img_data.shape[1]
        flag = 1
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >=1024:
        rate = 1024.0 / Img_data.shape[0]
        flag = 1

    if k%100==0:
        print (img_test[k],Img_data.shape)

    Img_data = cv2.resize(Img_data, (0, 0), fx=rate, fy=rate)

    if Img_data.shape[0]<min_y:
        min_y = Img_data.shape[0]
        #print(img_test[k])

    if Img_data.shape[1]<min_x:
        min_x = Img_data.shape[1]
    #print(min_y,min_x,rate)

    patch_x = Img_data.shape[1]/2
    patch_y = Img_data.shape[0]/2
    Gt_data = Gt_data['annPoints']

    Gt_data = Gt_data * rate


    density_map = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            density_map[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
    kpoint = density_map.copy()
    density_map = gaussian_filter(density_map, 6)


    #print(density_map.shape, Img_data.shape, img_train[k])
    new_img_path = (save_test_img_path + img_test[k])

    mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images','gt_show_density')
    h5_path = save_test_gt_path + img_test[k].replace('.jpg','.h5')



    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
    # pt2d = np.zeros(k.shape,dtype= np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1]) / 2
        sigma_map[pt[1], pt[0]] = sigma

    with h5py.File(h5_path, 'w') as hf:
        hf['density_map'] = density_map
        hf['kpoint'] = kpoint
        hf['sigma_map'] = sigma_map


    cv2.imwrite(new_img_path, Img_data)
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,2)
    cv2.imwrite(gt_show_path, density_map)
