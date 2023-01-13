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
for tl in train_set:      # checking if the data is iterable
  i+= 1
  print(i)
  break
print(len(train_set))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")        # Use CUDA when available and accessed by .to(device)

#####       CREATING MY DATATSET WHICH RETURNS - image , image_label , random int(in form of one hot encoded) , labels for sum

class MyDataset(Dataset):
  def __init__(self, train_set):
    self.data = train_set
  
  def __getitem__(self, index):
    random_num = random.randint(0,9)
    one_hot = np.zeros(10)
    one_hot[random_num] = 1
    one_hot = torch.tensor(one_hot)
    return self.data[index][0], self.data[index][1], one_hot , self.data[index][1] + random_num 

  def __len__(self):
    return len(self.data)

myData = MyDataset(train_set)

i = 0 
for m in myData:            # iterating to see the contents
  print(len(m) , m[0].shape, m[1], m[2].shape, m[2], m[3])
  i+= 1
  if i == 10:
    break
    
    ##      accessing the data and visualising it
sample = next(iter(myData))
image,image_label, random_num,final_label = sample
plt.imshow(image.squeeze(), cmap = 'gray')
print(random_num, image_label, final_label)

## loading the test batch data
train_loader = torch.utils.data.DataLoader(
    myData
    ,batch_size = 30
    ,shuffle = True
)

## Visualising the test batch data

sample = next(iter(train_loader))
images,image_label, random_numbers, final_labels = sample
print(random_numbers, image_label, final_labels)
print(images.shape , final_labels.shape)
grid = torchvision.utils.make_grid(images, nrow = 20)
plt.figure(figsize = (15,15))
plt.imshow(np.transpose(grid,(1,2,0)))

##          MAKING MY NETWORK

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 8 , kernel_size = 3)# input 28*28, output 26*26
    self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = 3)#input 26 , output 24
    self.conv3 = nn.Conv2d(in_channels = 16 ,out_channels = 32 , kernel_size = 3)#input 24, output 22
    self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1) # 22*22-> 22*22
    self.pool1 = nn.MaxPool2d(2,2) # 22*22 -> 11*11
    self.conv5 = nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = 3)# 11*11-> 9*9
    self.conv6 = nn.Conv2d(in_channels = 16 ,out_channels = 32 , kernel_size = 3)#9*9->7*7
    self.conv7 = nn.Conv2d(in_channels = 32 , out_channels = 8, kernel_size = 1)#7*7->7*7
    self.fc1 = nn.Linear(in_features = 8 * 7 * 7, out_features = 128)
    self.fc2 = nn.Linear(in_features = 128 , out_features = 64)
    self.out1 = nn.Linear(in_features = 64, out_features = 10)
    self.out2 = nn.Linear(in_features = 20 , out_features = 19)

  def forward(self,t1, t2):
    x = t1
    y = t2
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    x = F.max_pool2d(x, kernel_size = 2, stride = 2)

    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))
    
    x = x.reshape(-1, 8*7*7)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out1(x)
    x = F.softmax(x , dim = 1 )      

   
    y = torch.cat((x,y), dim = 1)
    y = self.out2(y)
    y = F.softmax(y , dim = 1)

    return x , y

network = MyNetwork()
    
## testing the network on an image making it fake batched
sample = next(iter(myData))
image,image_label, random_num,final_label = sample
pred1, pred2  = network(image.unsqueeze(0).to(torch.float32), random_num.unsqueeze(0).to(torch.float32))
print(pred1, pred2 , image_label , final_label)

## testing it on a 1 batch of 40 images
train_loader = torch.utils.data.DataLoader(
    myData
    ,batch_size = 40
    ,shuffle = True
)
batch = next(iter(train_loader))
images, batch_image_labels , random_num, final_label = batch
preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
print(preds1.shape , preds2.shape)
print(preds1,preds2)
def get_num_correct(preds, labels):
  return preds.argmax(dim = 1).eq(labels).sum().item()

## checking the accuracy of network by applying it on the same batch AGAIN AND AGAIN and seeing it improve before looping it
preds1.argmax(dim = 1).eq(batch_image_labels)
preds2.argmax(dim = 1).eq(final_label)
get_num_correct(preds1, batch_image_labels)
get_num_correct(preds2, final_label)

optimiser = optim.Adam(network.parameters(), lr = 0.01)

preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
loss1 = F.cross_entropy(preds1,batch_image_labels)
loss2 = F.cross_entropy(preds2, final_label)
loss = loss1 * loss2
print(loss1.item(), loss2.item(), get_num_correct(preds1,batch_image_labels),get_num_correct(preds2, final_label))
loss.backward()
optimiser.step()
optimiser.zero_grad()

preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
loss1 = F.cross_entropy(preds1,batch_image_labels)
loss2 = F.cross_entropy(preds2, final_label)
loss = loss1 * loss2
print(loss1.item(), loss2.item(), get_num_correct(preds1,batch_image_labels),get_num_correct(preds2, final_label))
loss.backward()
optimiser.step()
optimiser.zero_grad()

preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
loss1 = F.cross_entropy(preds1,batch_image_labels)
loss2 = F.cross_entropy(preds2, final_label)
loss = loss1 * loss2
print(loss1.item(), loss2.item(), get_num_correct(preds1,batch_image_labels),get_num_correct(preds2, final_label))
loss.backward()
optimiser.step()
optimiser.zero_grad()

preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
loss1 = F.cross_entropy(preds1,batch_image_labels)
loss2 = F.cross_entropy(preds2, final_label)
loss = loss1 * loss2
print(loss1.item(), loss2.item(), get_num_correct(preds1,batch_image_labels),get_num_correct(preds2, final_label))
loss.backward()
optimiser.step()
optimiser.zero_grad()

preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
loss1 = F.cross_entropy(preds1,batch_image_labels)
loss2 = F.cross_entropy(preds2, final_label)
loss = loss1 * loss2
print(loss1.item(), loss2.item(), get_num_correct(preds1,batch_image_labels),get_num_correct(preds2, final_label))
loss.backward()
optimiser.step()
optimiser.zero_grad()

#########           MAIN CODE           ######          TRAINING            #######
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
network = MyNetwork().to(device)

train_loader = torch.utils.data.DataLoader(myData , batch_size = 32, shuffle = True)
optimiser = optim.Adam(network.parameters() , lr = 0.001)

total_loss = 0
total_correct = 0

i = 0 
for epoch in range(10):
  
  total_loss1 = 0
  total_loss2 = 0
  total_correct1 = 0
  total_correct2 = 0

  for batch in train_loader:
    images, batch_image_labels ,random_num,final_label = batch
    images, batch_image_labels = images.to(device) , batch_image_labels.to(device)
    random_num, final_label = random_num.to(device), final_label.to(device)
    preds1 , preds2 = network(images.to(torch.float32), random_num.to(torch.float32))
    loss1 = F.cross_entropy(preds1,batch_image_labels)
    loss2 = F.cross_entropy(preds2, final_label)
    loss = loss1 + loss2
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    total_loss1 += loss1.item()
    total_loss2 += loss2.item()
    total_correct1 += get_num_correct(preds1, batch_image_labels)
    total_correct2 += get_num_correct(preds2, final_label)

  print(
      "epoch:", epoch,
      "total correct for image :", total_correct1,
      "total correct for sum :", total_correct2,
      "loss for image:",total_loss1,
      "loss for sum:",total_loss2
  )
