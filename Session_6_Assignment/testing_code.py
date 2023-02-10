from tqdm import tqdm
import torch



class test:
    def __init__(self):

        self.test_losses = []
        self.test_accuracies= []      

    def execute(self, model, testloader, device, criterion):
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