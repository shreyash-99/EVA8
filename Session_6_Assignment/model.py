from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary import summary


def get_model(dropout_value = 0.07):
    return Net(dropout_value)


def get_model_summary(model, device, image_dimension):
    model = model.to(device)
    summary(model, input_size = image_dimension)


class Net(nn.Module):
    def __init__(self, dropout_value):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size =(3,3), padding = 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #32 -> 32
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = (3,3), padding = 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #32->32
        # )
        # self.conv2_5 = nn.Sequential(
        #     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1 ),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value),
        #     nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1))
        # )
        # self.transition1 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,out_channels = 32,kernel_size = (3,3), dilation = 8)
        # )
        
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels = 32,out_channels =  64, kernel_size = (3,3), padding = 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #16 -> 16
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = (3,3), padding = 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #16->16
        # )
        # self.transition2 = nn.Sequential(
        #     nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = (3,3), dilation = 4)
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,out_channels =  128,kernel_size = (3,3), padding = 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #8 -> 8
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = (3,3), padding = 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_value) #8->8
        # )
        # self.transition3 = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = (3,3), dilation = 2)
        # )

        # self.gap = nn.Sequential(
        #     nn.AvgPool2d(kernel_size = 4)
        # )
        # self.fc = nn.Linear(64,10)


    def forward(self, x):
        # x = self.transition1(self.conv2(self.conv1(x)))
        # x = self.transition2(self.conv4(self.conv3(x)))
        # x = self.transition3(self.conv6(self.conv5(x)))
        # x = self.gap(x)
        # x = x.view(-1,64)
        # x = self.fc(x)



        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        return x