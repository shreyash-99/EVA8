
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class train:
    def __init__(self):
        
        self.train_loss = []
        self.train_acc = []
        self.train_accuracies = []
        self.train_losses = []
        

    def execute_training(self, model, trainloader, device, optimiser, criterion, epoch):
        model.train()
        if epoch == 0:
            self.train_accuracies = []
            self.train_losses = []
        train_loss = 0
        correct = 0
        #total = 0
        processed = 0
        pbar = tqdm(trainloader)

        for batch_idx, (data, target) in enumerate(pbar):
            #get samples
            data, target = data.to(device), target.to(device)

            #init
            optimiser.zero_grad()

            #prediction
            y_pred = model(data)

            #calculate loss
            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)  


            pred = y_pred.argmax(dim = 1, keepdim = True) # gets the index of the max log-probabilirty
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

            self.train_accuracies[epoch] = self.train_acc[-1]
            self.train_losses[epoch] = self.train_loss[-1]
        return 
            

