
import torch                   #PyTorch base libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2


class Albumentation_cifar_Dataset(Dataset):
  def __init__(self, image_list, train= True):
      self.image_list = image_list
      self.aug = A.Compose({
        A.PadIfNeeded(4),
        A.RandomCrop(32,32),
        # A.ShiftScaleRotate(),
        A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None),
        A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        # A.ToGray()
      })

      self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      })
      self.train = train
        
  def __len__(self):
      return (len(self.image_list))

  def __getitem__(self, i):
      
      image, label = self.image_list[i]
      
      if self.train:
        #apply augmentation only for training
        image = self.aug(image=np.array(image))['image']
      else:
        image = self.norm(image=np.array(image))['image']
      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      return torch.tensor(image, dtype=torch.float), label


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



def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))



def visualize_augmented_images(loader, classes, cols = 5, rows = 4):
    images, labels = next(iter(loader))

    sample_size=cols * rows

    images = images[0:sample_size]
    labels = labels[0:sample_size]

    fig = plt.figure(figsize=(10, 10))

    # Show images
    for idx in np.arange(len(labels.numpy())):
        ax = fig.add_subplot(cols, rows, idx+1, xticks=[], yticks=[])
        npimg = unnormalize(images[idx])
        ax.imshow(npimg, cmap='gray')
        ax.set_title("Label={}".format(classes[labels[idx]]))

    fig.tight_layout()  
    plt.show()

def compute_accuracy_graph(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Epoch vs Accuracy")
    plt.legend()

def compute_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Epoch vs Accuracy")
    plt.legend()

