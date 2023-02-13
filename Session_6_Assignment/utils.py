# import necessary libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import numpy as np
import cv2
import os

# from data_loader import unnormalize

def plot_misclassified_images(model, test_loader, device , cols = 5 ,rows = 4):
    all_misclassified_images = []

    model.eval()

    with torch.no_grad():
        for data , labels in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    all_misclassified_images.append({'image': data[i], 'predicted_class': pred[i], 'correct_class': target[i]})

    fig = plt.figure(figsize=(15,5))
    for i in range(cols * rows):
        sub = fig.add_subplot(cols, rows, i+1)
        misclassified_image = all_misclassified_images[i]
        plt.imshow(misclassified_image['image'].cpu().numpy().squeeze(), cmap='gray', interpolation='none')
        sub.set_title("Correct class: {}\nPredicted class: {}".format(misclassified_image['correct_class'], misclassified_image['predicted_class']))
    plt.tight_layout()
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

