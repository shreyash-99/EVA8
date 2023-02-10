
import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

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

