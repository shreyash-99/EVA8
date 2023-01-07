import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np

import random

# ACCESSING THE MNIST DATASET

train_set = torchvision.datasets.MNIST(
    root = './data'
    ,train = True
    ,download = True
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

i =0 
for tl in train_set:      #CHECKING IF IT IS ITERABLE
  i+= 1
  print(i)
  break
