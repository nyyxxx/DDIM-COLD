import math
import os
import numpy as np
import sys
sys.path.append('/home/ailanyy1/anaconda3/lib/python3.8/site-packages')
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
# import torchvision
import torchvision.transforms.functional as F
# print("Torchvision Version: ",torchvision.__version__)
# print(torch.__version__)
# import torchvision.transforms.Interpolation as I
import random


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DiffusionDataset(Dataset): 
    '''
    your folder should have following structrues:
    
    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32],
                 max_step=2000):

        self.root = root
        self.width, self.height = imgSize
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        self.max_step = max_step
    def __getitem__(self, index,t=None):
        index = random.randint(0,9)
        path = os.path.join(self.root, self.imgList[int(index)])
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height))
        img = img*2 -1
        if t is None:
            t = int(np.random.randint(self.max_step, size=1))
        alpha = 1 - math.sqrt((t+1)/self.max_step)
        noise = torch.normal(0,1,img.shape)
        noisy_img = math.sqrt(alpha)*img+math.sqrt(1-alpha)*noise
        return noisy_img,img,t

    def __len__(self):
        return len(self.imgList)

class ColdDownSampleDataset(Dataset): 
    '''
    your folder should have following structrues:

    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32]):
        self.root = root
        self.width, self.height = imgSize
        assert self.width ==  self.height,'downsample dataset reqiure square images'
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        self.max_step = np.log2(imgSize[0])

    def get_t(self,img,t):
        target_size = math.floor(self.width / t )  #nyy change the process
        noisy_img = F.resize(img,(target_size,target_size),F.InterpolationMode.NEAREST)
        noisy_img = F.resize(noisy_img,(self.width,self.height),F.InterpolationMode.NEAREST)  #TAG downsample    
        return noisy_img  
    def __getitem__(self, index,t=None):
        #index = random.randint(0,9)
        path = os.path.join(self.root, self.imgList[int(index)])
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height))
        img = img*2 -1
        if t is None:
            t = int(np.random.randint(self.max_step, size=1))+1
            # print('t='+str(t))
        noisy_t = self.get_t(img,2**t)
        noisy_t_1 = self.get_t(img,2**(t-1))
        #return noisy_img,img,t
        return noisy_t,noisy_t_1,t   #TAG return groundtruth noisy image   
    
class ColdDownSampleDataset_au(Dataset): #author's cold diff
    '''
    your folder should have following structrues:

    root
        --1.jpg
        --2.jpg
        --...
    '''
    def __init__(self,
                 root: str,
                 imgSize=[32,32]):
        self.root = root
        self.width, self.height = imgSize
        assert self.width ==  self.height,'downsample dataset reqiure square images'
        self.imgList = os.listdir(root)
        self.list = os.listdir(root)
        self.max_step = np.log2(imgSize[0])

    def get_t(self,img,t):
        target_size = math.floor(self.width / t )  #nyy change the process
        noisy_img = F.resize(img,(target_size,target_size),F.InterpolationMode.NEAREST)
        noisy_img = F.resize(noisy_img,(self.width,self.height),F.InterpolationMode.NEAREST)  
        return noisy_img  
    def __getitem__(self, index,t=None):
        #index = random.randint(0,9)
        path = os.path.join(self.root, self.imgList[int(index)])
        img = pil_loader(path)
        img = F.to_tensor(img)
        img = F.resize(img,(self.width,self.height))
        img = img*2 -1
        if t is None:
            t = int(np.random.randint(self.max_step, size=1))+1
            # print('t='+str(t))
        noisy_img = self.get_t(img,2**t)
        # noisy_t_1 = self.get_t(img,2**(t-1))
        return noisy_img,img,t  #TAG return img_0 

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = ColdDownSampleDataset("/data3/yuyan/DDIM-COLD/OxfordFlowers/train",[64,64])
    for i in range(1,7,1):   # t=1...6  (64=2^6)  get img_t and img_t-1
        print(i)
        img,noisy_img,t = dataset.__getitem__(0,i)  #nyy change the process
        img = (img + 1)/2
        noisy_img = (noisy_img + 1)/2
        img,noisy_img = F.to_pil_image(img),F.to_pil_image(noisy_img)
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax2.imshow(noisy_img)
        plt.show()
        plt.savefig('c1' + '_labeled.jpg')