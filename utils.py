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

def imlabshow(img,mode='train',name=None):
  npimg=img.cpu().numpy()
  npimg=npimg/2 +0.5
  np_lab_img=npimg.transpose(1,2,0)
  np_lab_img*=255
  np_rgb_img=cv2.cvtColor(np_lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
  plt.imshow(np_rgb_img)
  plt.show()
  if mode=='test':
    np_rgb_img=cv2.cvtColor(np_rgb_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(config.dir+'dataset/test/results/'+name+'_Original',np_rgb_img)

  #np_rgb_img=cv2.cvtColor(np_rgb_img,cv2.COLOR_RGB2BGR)
  #cv2_imshow(np_rgb_img)

def imfakeshow(img,mode='train',name=None):
  npimg=img.detach().cpu().numpy()
  npimg=npimg/2 +0.5
  np_lab_img=npimg.transpose(1,2,0)
  np_lab_img*=255
  np_rgb_img=cv2.cvtColor(np_lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
  plt.imshow(np_rgb_img)
  plt.show()
  if mode=='test':
    np_rgb_img=cv2.cvtColor(np_rgb_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(config.dir+'dataset/test/results/'+name+'_Fake',np_rgb_img)

def imgrayshow(img,mode='train',name=None):
  npimg=img.detach().cpu().numpy()
  npimg=npimg/2 + 0.5
  np_lab_img=npimg.transpose(1,2,0)
  np_lab_img*=255
  cv2_imshow(np_lab_img)
  if mode=='test':
    cv2.imwrite(config.dir+'dataset/test/results/'+name+'_GrayScale',np_lab_img)

def save_weights(state,step_no):
  weight=glob.glob(config.dir+'weights/latest*')
  assert len(weight)<=1, "Multiple weights file, delete others."
  if weight:
    os.remove(weight[0])
  print("Saving weights as latest_colorize_weights_"+str(step_no))
  torch.save(state,config.dir+"weights/latest_colorize_weights_"+str(step_no)+".pth.tar")

def save_loss_hist(loss_d,loss_g):
  loss_g_hist=glob.glob(config.dir+'Gen_loss_*')
  loss_d_hist=glob.glob(config.dir+'Dis_loss_*')
  assert len(loss_g_hist)<=1, "Multiple files of Gen History"
  assert len(loss_d_hist)<=1, "Multiple files of Dis History"
  if loss_g_hist:
    os.remove(loss_g_hist[0])
  if loss_d_hist:
    os.remove(loss_d_hist[0])
  open_file = open(config.dir+"Gen_loss_hist.pkl", "wb")
  pickle.dump(loss_g, open_file)
  open_file.close()
  open_file = open(config.dir+"Dis_loss_hist.pkl", "wb")
  pickle.dump(loss_d, open_file)
  open_file.close()

def stitch_images(grayscale, original, pred):
    gap = 30
    width, height = original[:, :, 0].shape
    img = Image.new('RGB', (width* 3 + gap * 2, height),color=(255,255,255))

    grayscale = np.array(grayscale).squeeze()
    original = np.array(original)
    pred = np.array(pred)

    im1 = Image.fromarray(grayscale)
    im2 = Image.fromarray(original)
    im3 = Image.fromarray((pred).astype(np.uint8))
    img.paste(im1, (0,0))
    img.paste(im2, (width+gap, 0))
    img.paste(im3, (width + width+gap+gap, 0))

    return img