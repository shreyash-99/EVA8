
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import albumentations as A

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2


def train_loader():
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Normalize((0.4914, 0.4822, 0.4471), (0.2469, 0.2433, 0.2615)),
            A.Cutout(num_holes = 1, max_h_size = 16, max_w_size = 16, fill_value = 0.5, always_apply = False, p = 0.5)
        ]
    )

def info_about_dataset(exp, train = True):
    exp_data = exp.data

    # Calculate the mean and std for normalization
    if train:
        print("[Train")
    else :
        print("Test")
    print(' - Numpy Shape:', exp_data.shape)
    print(' - min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - var:', np.var(exp_data, axis=(0,1,2)) / 255.)