# import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

from data_loader import unnormalize

def plot_misclassified_images(model, test_loader, classes, device , cols = 5 ,rows = 4):
    all_misclassified_images = []
    model.eval()
    with torch.no_grad():
        for data , target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    all_misclassified_images.append({'image': data[i], 'predicted_class': classes[pred[i]], 'correct_class': classes[target[i]]})

    fig = plt.figure(figsize=(10,10))
    for i in range(cols * rows):
        sub = fig.add_subplot(cols, rows, i+1)
        misclassified_image = all_misclassified_images[i]
        plt.imshow(misclassified_image['image'].cpu().permute(1,2,0).numpy().squeeze(), cmap='gray', interpolation='none')
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

