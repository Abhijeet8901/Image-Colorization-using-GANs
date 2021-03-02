import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import sampler
import torchvision.datasets as dset
from torch.autograd import Variable

import os
import time
import pickle
import cv2
import numpy as np
import glob
from PIL import Image
import shutil

from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchsummary import summary

from itertools import product
from math import sqrt

dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Places365Loader(data.Dataset):

    def __init__(self,mode='train'):
      
      self.mode=mode
      self.cfg=config
      self.data_root='data/'
      if self.mode=='train':
        self.data_path=self.data_root+'train/*.*'
        self.img_path_list=glob.glob(self.data_path)

      elif self.mode=='val':
        self.data_path=config.dir+'dataset/val/*.*'
        self.img_path_list=glob.glob(self.data_path)
      
      elif self.mode=='test':
        self.data_path=config.dir+'dataset/test/'+self.cfg.test_img_name

    def __getitem__(self,index):

      if self.mode=='test':

        image=cv2.imread(self.data_path)
        lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        lab_image= cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)
        lab_image=lab_image.astype(np.float64)
        lab_image/=255.0
        lab_image=torch.from_numpy(lab_image.transpose(2,0,1))
        l_image=lab_image[0,:,:].unsqueeze(0)
        c_image=lab_image[1:,:,:]
        mean=torch.Tensor([0.5])
        l_image=l_image-mean.expand_as(l_image)
        c_image=c_image-mean.expand_as(c_image)
        l_image=2*l_image
        c_image=2*c_image
        gray_image=l_image
        lab_image=torch.cat([l_image,c_image],0)
        image=torch.from_numpy(image.transpose(2,0,1))
        return lab_image,gray_image,image

      if self.mode=='val':
        image=cv2.imread(self.img_path_list[index])
        lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        lab_image= cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)
        lab_image=lab_image.astype(np.float64)
        lab_image/=255.0
        lab_image=torch.from_numpy(lab_image.transpose(2,0,1))
        l_image=lab_image[0,:,:].unsqueeze(0)
        c_image=lab_image[1:,:,:]
        mean=torch.Tensor([0.5])
        l_image=l_image-mean.expand_as(l_image)
        c_image=c_image-mean.expand_as(c_image)
        l_image=2*l_image
        c_image=2*c_image
        gray_image=l_image
        lab_image=torch.cat([l_image,c_image],0)
        image=torch.from_numpy(image.transpose(2,0,1))
        return lab_image,gray_image,image

      if self.mode=='train':
        image=cv2.imread(self.img_path_list[index])
        lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        lab_image= cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)
        lab_image=lab_image.astype(np.float64)
        lab_image/=255.0
        lab_image=torch.from_numpy(lab_image.transpose(2,0,1))
        l_image=lab_image[0,:,:].unsqueeze(0)
        c_image=lab_image[1:,:,:]
        mean=torch.Tensor([0.5])
        l_image=l_image-mean.expand_as(l_image)
        c_image=c_image-mean.expand_as(c_image)
        l_image=2*l_image
        c_image=2*c_image
        gray_image=l_image
        lab_image=torch.cat([l_image,c_image],0)
        image=torch.from_numpy(image.transpose(2,0,1))
        return lab_image,gray_image,image
    
    def __len__(self):

      if self.mode=='test':
        return 1
      else:
        return len(self.img_path_list)


def train_collate(batch):

  lab_list,gray_list,img_list=[],[],[]
  for i,sample in enumerate(batch):
    lab_list.append(sample[0])
    gray_list.append(sample[1])
    img_list.append(sample[2])
  lab_imgs=torch.stack(lab_list)
  gray_imgs=torch.stack(gray_list)
  imgs=torch.stack(img_list)
  return lab_imgs,gray_imgs,imgs


def val_collate(batch):

  lab_list,gray_list,img_list=[],[],[]
  for i,sample in enumerate(batch):
    lab_list.append(sample[0])
    gray_list.append(sample[1])
    img_list.append(sample[2])
  lab_imgs=torch.stack(lab_list)
  gray_imgs=torch.stack(gray_list)
  imgs=torch.stack(img_list)
  return lab_imgs,gray_imgs,imgs


def test_collate(batch):

  lab_list,gray_list,img_list=[],[],[]
  for i,sample in enumerate(batch):
    lab_list.append(sample[0])
    gray_list.append(sample[1])
    img_list.append(sample[2])
  lab_imgs=torch.stack(lab_list)
  gray_imgs=torch.stack(gray_list)
  imgs=torch.stack(img_list)
  return lab_imgs,gray_imgs,imgs