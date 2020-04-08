import torch
from torch.utils.data import Dataset
import os
from image import *

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img,target,kpoint,sigma_map= load_data(img_path,self.train)

        '''data augmention'''
        if self.train==True:
            if random.random() > 0.5:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.fliplr(kpoint)
                sigma_map = np.fliplr(sigma_map)

            if random.random() > 0.5:
                proportion = random.uniform(0.004, 0.015)
                width, height = img.size[0], img.size[1]
                num = int(height * width * proportion)
                for i in range(num):
                    w = random.randint(0, width - 1)
                    h = random.randint(0, height - 1)
                    if random.randint(0, 1) == 0:
                        img.putpixel((w, h), (0, 0, 0))
                    else:
                        img.putpixel((w, h), (255, 255, 255))

        target = target.copy()
        kpoint = kpoint.copy()
        img = img.copy()
        sigma_map = sigma_map.copy()

        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(target).cuda()

        return img,target,kpoint,fname,sigma_map
