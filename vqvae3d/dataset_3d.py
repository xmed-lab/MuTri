import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np
import SimpleITK as sitk

def crop(data, h=192, l=40, train=True):
    idx = np.random.randint(0,3, 1)[0]
    if idx == 0:
        sh, sw, sd = h, h, l
    elif idx == 1:
        sh, sw, sd = h, l, h
    else:
        sh, sw, sd = l, h, h
    B = 512
    data = data[:,:,:2*B]
    H, W, D = data.shape
    max_spacing = min(H/sh, W/sw, D/sd, 2.5)   # 2.0 / 0.8 = 2.5
    current_spacing = np.random.rand(1)[0] * (max_spacing - 1) + 1
    spacing = current_spacing * 0.8
    max_h, max_w, max_d = int(current_spacing*sh), int(current_spacing*sw), int(current_spacing*sd)

    bh = np.random.randint(0, H-max_h+1, 1)[0]
    bw = np.random.randint(0, W-max_w+1, 1)[0]
    bd = np.random.randint(0, D-max_d+1, 1)[0]
    #print(H,W,D,(bh, max_h), (bw, max_w), (bd, max_d))

    crop_data = data[bh:bh+max_h, bw:bw+max_w, bd:bd+max_d].unsqueeze(0).unsqueeze(0)

    crop_data = F.interpolate(crop_data, size=(sh, sw, sd), mode='trilinear')[0]

    absolute_bh = (bh - H//2) / B; absolute_bw = (bw - W//2) / B; absolute_bd = (bd - D//2) / B

    return crop_data, [spacing, absolute_bh, absolute_bw, absolute_bd]


class SliceSet(Dataset):
    
    def __init__(self, txt, h=192, l=40, train=True):
        
        self.train = train

        f = open(txt, "r")
        self.data = []
        self.h = h
        self.l = l
        print("high {} low {} resolution".format(self.h, self.l))
        files = f.readlines()
        for file in files:
            self.data.append(file.replace('\n', ''))
         
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        data = np.load(self.data[index])
        data = torch.from_numpy(data)
        info = [0.8, 0.8, 0.8]
        if self.train:
            data, info = crop(data, self.h, self.l)
            #print(data.size())
            #imgs = sitk.GetImageFromArray(data[0].numpy())
            #sitk.WriteImage(imgs, "test.nii.gz")
        data = torch.clamp(data, min=-1000., max=1000.)
        info = torch.Tensor(info)
        data = (data + 1000.) / 2000.
        return data, info


def get_dataset(txt='./', h=192, l=40, train=True):
    train_set = SliceSet(txt, h=h, l=l, train=train)
    return train_set


if __name__ == '__main__':
    h = 160
    l = 40
    txt = "/nfs/scratch/whl/LungCA_0.8/data.txt"
    #txt = "/nfs/scratch/whl/PenWin_0.8/data.txt"
    dataset = get_dataset(txt, h=h, l=l)
    lis = [i for i in range(900)]
    random.shuffle(lis)
    data, info = dataset[lis[0]] 
    data = data * 2000. - 1000.
    imgs = sitk.GetImageFromArray(data[0].numpy())
    sitk.WriteImage(imgs, "test.nii.gz")
    print(info)
    
