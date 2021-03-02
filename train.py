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

from dataloader import *
from models import *
from utils import *

dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Gen_Model=Generator().cuda()
Dis_Model=Discriminator().cuda()
Gen_Model.train()
Dis_Model.train()

dataset=Places365Loader(config.mode)
optimizer_G=optim.Adam(Gen_Model.parameters(),lr=config.lr,betas=(0.5, 0.999))
optimizer_D=optim.Adam(Dis_Model.parameters(),lr=config.lr,betas=(0.5, 0.999))
train_loader=DataLoader(dataset,config.batch_size,shuffle=True,collate_fn=train_collate)
loss_g=[]
loss_d=[]
if config.resume:
  checkpoint=torch.load(config.save_path)
  Gen_Model.load_state_dict(checkpoint['state_dict_G'])
  Dis_Model.load_state_dict(checkpoint['state_dict_D'])
  optimizer_G.load_state_dict(checkpoint['optimizer_G'])
  optimizer_D.load_state_dict(checkpoint['optimizer_D'])
  open_file=open(config.dir+'Gen_loss_hist.pkl','rb')
  loss_g=pickle.load(open_file)
  open_file.close()
  open_file=open(config.dir+'Dis_loss_hist.pkl','rb')
  loss_d=pickle.load(open_file)
  open_file.close()
  print(f'\nResume training with \'{config.save_path}\'.\n')
else:
  Gen_Model.init_weights()
  Dis_Model.init_weights()

torch.autograd.set_detect_anomaly(True)
step=1
if config.resume:
  step=checkpoint['step']+1
training=True
D_BCE=nn.BCELoss()
G_BCE=nn.BCELoss()
L1=nn.L1Loss()
real_label=0.9
fake_label=0.0
time_last=time.time()
while training:
  for i,(lab_imgs,gray_imgs,imgs) in enumerate(train_loader):

      lab_imgs=Variable(lab_imgs.cuda().type(dtype))
      gray_imgs=Variable(gray_imgs.cuda().type(dtype))
      imgs=Variable(imgs.cuda().type(dtype))
      #### Update D Network ####
      Dis_Model.zero_grad()
      output=Dis_Model(lab_imgs)
      output=torch.squeeze(output,1)
      error_D_real=D_BCE(output,((real_label) * torch.ones(output.size(0))).cuda())

      fake_img=Gen_Model(gray_imgs).detach()
      output=Dis_Model(fake_img)
      output=torch.squeeze(output,1)
      error_D_fake=D_BCE(output,(torch.zeros(output.size(0))).cuda())

      error_D = error_D_real + error_D_fake
      loss_d.append(error_D)
      error_D.backward()

      optimizer_D.step()

      #### Update G Network ####

      Gen_Model.zero_grad()
      fake_img=Gen_Model(gray_imgs)
      output=Dis_Model(fake_img)
      output=torch.squeeze(output,1)
      error_G_GAN=G_BCE(output,(torch.ones(output.size(0))).cuda())
      error_G_L1=L1(fake_img.view(fake_img.size(0),-1),lab_imgs.view(lab_imgs.size(0),-1))

      error_G = error_G_GAN + config.lamb * error_G_L1

      error_G.backward()
      optimizer_G.step()
      loss_g.append(error_G)
      ##########
      this_time=time.time()
      if i%50==0:
        print("Batch No -",i,"Completed with time",this_time-time_last,".Dis Losses = ",error_D_real.item(),",",error_D_fake.item(),". GAN Losses = ",error_G_GAN.item(),",",error_G_L1.item())
        with torch.no_grad():
          imfakeshow(fake_img[0])
      time_last=time.time()


  print("Epoch ",step," done")
  state={'step':step,
         'state_dict_G':Gen_Model.state_dict(),
         'state_dict_D':Dis_Model.state_dict(),
         'optimizer_G':optimizer_G.state_dict(),
         'optimizer_D':optimizer_D.state_dict()}
  save_weights(state,step)
  save_loss_hist(loss_d,loss_g)
  plt.plot(np.array(loss_d), 'r')
  plt.show()
  plt.plot(np.array(loss_g), 'r')
  plt.show()
  step+=1

plt.plot(np.array(loss_d), 'r')

plt.plot(np.array(loss_g), 'r')