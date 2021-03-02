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

class Generator(nn.Module):

    def __init__(self):
      super(Generator,self).__init__()
      self.conv1=nn.Conv2d(1,64,3,stride=1,padding=1,bias=False)
      self.relu1=nn.LeakyReLU(0.2)

      self.conv2=nn.Conv2d(64,64,3,stride=2,padding=1,bias=False)
      self.bn2=nn.BatchNorm2d(64,momentum=0.5)
      self.relu2=nn.LeakyReLU(0.2)
      
      self.conv3=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False)
      self.bn3=nn.BatchNorm2d(128,momentum=0.5)
      self.relu3=nn.LeakyReLU(0.2)

      self.conv4=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False)
      self.bn4=nn.BatchNorm2d(256,momentum=0.5)
      self.relu4=nn.LeakyReLU(0.2)

      self.conv5=nn.Conv2d(256,512,3,stride=2,padding=1,bias=False)
      self.bn5=nn.BatchNorm2d(512,momentum=0.5)
      self.relu5=nn.LeakyReLU(0.2)

      self.conv6=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn6=nn.BatchNorm2d(512,momentum=0.5)
      self.relu6=nn.LeakyReLU(0.2)

      self.conv7=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn7=nn.BatchNorm2d(512,momentum=0.5)
      self.relu7=nn.LeakyReLU(0.2)

      self.conv8=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn8=nn.BatchNorm2d(512,momentum=0.5)
      self.relu8=nn.LeakyReLU(0.2)

      self.conv9=nn.ConvTranspose2d(512,512,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn9=nn.BatchNorm2d(512,momentum=0.5)
      self.relu9=nn.ReLU()
      
      self.conv10=nn.ConvTranspose2d(1024,512,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn10=nn.BatchNorm2d(512,momentum=0.5)
      self.relu10=nn.ReLU()

      self.conv11=nn.ConvTranspose2d(1024,512,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn11=nn.BatchNorm2d(512,momentum=0.5)
      self.relu11=nn.ReLU()

      self.conv12=nn.ConvTranspose2d(1024,256,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn12=nn.BatchNorm2d(256,momentum=0.5)
      self.relu12=nn.ReLU()

      self.conv13=nn.ConvTranspose2d(512,128,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn13=nn.BatchNorm2d(128,momentum=0.5)
      self.relu13=nn.ReLU()

      self.conv14=nn.ConvTranspose2d(256,64,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn14=nn.BatchNorm2d(64,momentum=0.5)
      self.relu14=nn.ReLU()

      self.conv15=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1,bias=False)
      self.bn15=nn.BatchNorm2d(64,momentum=0.5)
      self.relu15=nn.ReLU()
      
      self.conv16=nn.Conv2d(128,3,1,stride=1,bias=False)

    def forward(self,img):
      
      x1=self.relu1(self.conv1(img))

      x2=self.relu2(self.bn2(self.conv2(x1)))

      x3=self.relu3(self.bn3(self.conv3(x2)))

      x4=self.relu4(self.bn4(self.conv4(x3)))

      x5=self.relu5(self.bn5(self.conv5(x4)))

      x6=self.relu6(self.bn6(self.conv6(x5)))

      x7=self.relu7(self.bn7(self.conv7(x6)))

      x8=self.relu8(self.bn8(self.conv8(x7)))

      x9=self.relu9(self.bn9(self.conv9(x8)))
      x9=torch.cat([x7,x9],1)

      x10=self.relu10(self.bn10(self.conv10(x9)))
      x10=torch.cat([x6,x10],1)

      x11=self.relu11(self.bn11(self.conv11(x10)))
      x11=torch.cat([x5,x11],1)

      x12=self.relu12(self.bn12(self.conv12(x11)))
      x12=torch.cat([x4,x12],1)

      x13=self.relu13(self.bn13(self.conv13(x12)))
      x13=torch.cat([x3,x13],1)

      x14=self.relu14(self.bn14(self.conv14(x13)))
      x14=torch.cat([x2,x14],1)

      x15=self.relu15(self.bn15(self.conv15(x14)))
      x15=torch.cat([x1,x15],1)

      x16=self.conv16(x15)
      x16=torch.tanh(x16)

      return x16
    
    def init_weights(self):

      for name,module in self.named_modules():
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.ConvTranspose2d):
          nn.init.xavier_uniform_(module.weight.data)
          if module.bias is not None:
            module.bias.data.zero_()

class Discriminator(nn.Module):

    def __init__(self):
      super(Discriminator,self).__init__()
      
      self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
      self.relu1=nn.LeakyReLU(0.2)

      self.conv2=nn.Conv2d(64,64,3,stride=2,padding=1,bias=False)
      self.bn2=nn.BatchNorm2d(64,momentum=0.5)
      self.relu2=nn.LeakyReLU(0.2)

      self.conv3=nn.Conv2d(64,128,3,stride=2,padding=1,bias=False)
      self.bn3=nn.BatchNorm2d(128,momentum=0.5)
      self.relu3=nn.LeakyReLU(0.2)

      self.conv4=nn.Conv2d(128,256,3,stride=2,padding=1,bias=False)
      self.bn4=nn.BatchNorm2d(256,momentum=0.5)
      self.relu4=nn.LeakyReLU(0.2)

      self.conv5=nn.Conv2d(256,512,3,stride=2,padding=1,bias=False)
      self.bn5=nn.BatchNorm2d(512,momentum=0.5)
      self.relu5=nn.LeakyReLU(0.2)

      self.conv6=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn6=nn.BatchNorm2d(512,momentum=0.5)
      self.relu6=nn.LeakyReLU(0.2)

      self.conv7=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn7=nn.BatchNorm2d(512,momentum=0.5)
      self.relu7=nn.LeakyReLU(0.2)

      self.conv8=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
      self.bn8=nn.BatchNorm2d(512,momentum=0.5)
      self.relu8=nn.LeakyReLU(0.2)

      self.flat=nn.Flatten()
      self.fc1=nn.Linear(2048,100)
      self.relu9=nn.LeakyReLU(0.2)
      self.fc2=nn.Linear(100,1)
    
    def forward(self,img):

      x1=self.relu1(self.conv1(img))

      x2=self.relu2(self.bn2(self.conv2(x1)))

      x3=self.relu3(self.bn3(self.conv3(x2)))

      x4=self.relu4(self.bn4(self.conv4(x3)))

      x5=self.relu5(self.bn5(self.conv5(x4)))

      x6=self.relu6(self.bn6(self.conv6(x5)))

      x7=self.relu7(self.bn7(self.conv7(x6)))

      x8=self.relu8(self.bn8(self.conv8(x7)))

      x8=self.flat(x8)

      x9=self.relu9(self.fc1(x8))

      x10=torch.sigmoid(self.fc2(x9))

      return x10
    
    def init_weights(self):

      for name,module in self.named_modules():
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.ConvTranspose2d):
          nn.init.xavier_uniform_(module.weight.data)
          if module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module,nn.Linear):
          nn.init.xavier_uniform_(module.weight.data)