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

from dataloader import *
from models import *
from utils import *

####################### Sanity checks between training #####################
Gen_Model.eval()
BCE=nn.BCELoss()
Lloss=nn.L1Loss()
real_label=0.9
time_last=time.time()
with torch.no_grad():
  data_temp=Places365Loader('val')
  val_loader=DataLoader(data_temp,batch_size=1,shuffle=False,collate_fn=val_collate)
  for i,(lab_imgs,gray_imgs,imgs) in enumerate(val_loader):
    lab_imgs=lab_imgs.cuda().type(dtype)
    gray_imgs=gray_imgs.cuda().type(dtype)
    imgs=imgs.cuda().type(dtype)
    output=Dis_Model(lab_imgs)
    output=torch.squeeze(output,1)
    error_D_real=BCE(output,((real_label) * torch.ones(output.size(0))).cuda())

    fake_img=Gen_Model(gray_imgs)
    output=Dis_Model(fake_img)
    output=torch.squeeze(output,1)
    error_D_fake=BCE(output,(torch.zeros(output.size(0))).cuda())

    error_D = error_D_real + error_D_fake


    fake_img=Gen_Model(gray_imgs)
    output=Dis_Model(fake_img)
    output=torch.squeeze(output,1)
    error_G_GAN=BCE(output,(torch.ones(output.size(0))).cuda())
    error_G_L1=Lloss(fake_img.view(fake_img.size(0),-1),lab_imgs.view(lab_imgs.size(0),-1))

    error_G =error_G_GAN + config.lamb * error_G_L1

    this_time=time.time()
    print("Batch No -",i,"Completed with time",this_time-time_last,".Dis Losses = ",error_D_real.item(),",",error_D_fake.item(),". GAN Losses = ",error_G_GAN.item(),",",error_G_L1.item())
    time_last=time.time()
    for j in range(lab_imgs.size(0)):
      imgrayshow(gray_imgs[0])
      imlabshow(lab_imgs[0])
      imfakeshow(fake_img[0])