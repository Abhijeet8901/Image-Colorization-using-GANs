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

################# Test Time ######################
model=Generator().to(device)
model.eval()
checkpoint=torch.load(config.save_path)
model.load_state_dict(checkpoint['state_dict_G'])
config.test_img_name="test_3.jpg"
with torch.no_grad():
  test_data=Places365Loader('test')
  test_loader=DataLoader(test_data,config.batch_size,shuffle=False,collate_fn=test_collate)
  for i,(lab_imgs,gray_imgs,imgs) in enumerate(test_loader):
    lab_imgs=lab_imgs.cuda().type(dtype)
    gray_imgs=gray_imgs.cuda().type(dtype)
    imgs=imgs.cuda().type(dtype)
    fake_img=model(gray_imgs)
    for j in range(lab_imgs.size(0)):
      print("GrayScale Image - ")
      imgrayshow(gray_imgs[0],'test',config.test_img_name)
      print("Original Image - ")
      imlabshow(lab_imgs[0],'test',config.test_img_name)
      print("Model Output Image - ")
      imfakeshow(fake_img[0],'test',config.test_img_name)
