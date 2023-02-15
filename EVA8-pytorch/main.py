
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import matplotlib.pyplot as plt

class train:
    def __init__(self):

        self.train_accuracies = []
        self.train_losses = []

    def execute(self, model, trainloader, device, optimiser, criterion, epoch):
        model.train()
        # if epoch == 0:
        #     self.train_accuracies = []
        #     self.train_losses = []
        correct = 0
        processed = 0
        pbar = tqdm(trainloader)
        training_loss_this_epoch = 0
        training_acc_this_epoch = 0

        print("EPOCH: {} (LR: {})".format(epoch, optimiser.param_groups[0]['lr']))

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
            training_loss_this_epoch += loss.item()
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


class test:
    def __init__(self):

        self.test_losses = []
        self.test_accuracies= []      

    def execute(self, model, testloader, device, criterion, epoch):
        model.eval()
        loss_this_epoch = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss_this_epoch += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        loss_this_epoch /= len(testloader.dataset)
        self.test_losses.append(loss_this_epoch)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss_this_epoch, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

        # Save.
        self.test_accuracies.append(100. * correct / len(testloader.dataset))  
        self.test_losses.append(loss_this_epoch)     

    def compute_accuracy_graph(self):
        plt.plot(self.test_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title("Epoch vs Accuracy while Testing")
        plt.legend()

    def compute_loss_graph(self):
        plt.plot(self.test_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Epoch vs Accuracy while Testing")
        plt.legend()     


def train_network(model, device, trainloader, testloader, EPOCHS, optimizer, criterion, scheduler  = None):
    criterion = criterion
    optimizer = optimizer
    scheduler = scheduler

    trainObj = train()
    testObj = test()

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        trainObj.execute(model, trainloader, device, optimizer, criterion, epoch)
        testObj.execute(model, testloader, device, criterion, epoch)
        scheduler.step()

    print('Finished Training')

    return trainObj, testObj

def calClassAccuracy(net, dataloader, classes, device):  
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))