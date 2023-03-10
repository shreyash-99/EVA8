
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class train:
    def __init__(self):

        self.train_accuracies = []
        self.train_losses = []

    def execute_training(self, model, trainloader, device, optimiser, criterion, scheduler, epoch):
        model.train()
        # if epoch == 0:
        #     self.train_accuracies = []
        #     self.train_losses = []
        correct = 0
        processed = 0
        pbar = tqdm(trainloader)
        training_loss_this_epoch = 0
        training_acc_this_epoch = 0

        for batch_idx, (data, target) in enumerate(pbar):
            #get samples
            data, target = data.to(device), target.to(device)

            #init
            optimiser.zero_grad()

            #prediction
            y_pred = model(data)

            #calculate loss
            loss = criterion(y_pred, target)


            # Backpropagation
            loss.backward()
            optimiser.step()

            # train_loss += loss.item()

            pred = y_pred.argmax(dim = 1, keepdim = True) # gets the index of the max log-probabilirty
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            training_acc_this_epoch += 100. * pred.eq(target.view_as(pred)).sum().item() / len(data)
            training_loss_this_epoch += loss
        self.train_accuracies.append(100. * correct / processed)
        self.train_losses.append(training_loss_this_epoch)
    
    def compute_accuracy_graph(self):
        plt.plot(self.train_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title("Epoch vs Accuracy while Training")
        plt.legend()

    def compute_loss_graph(self):
        plt.plot(self.train_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Epoch vs Accuracy while Training")
        plt.legend()
