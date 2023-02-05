from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms



dropout_value = 0.07

def get_model(normalisation_type):
    if(normalisation_type == "BN"):
        return BN_model()
    elif(normalisation_type == "LN"):
        return LN_model()
    elif(normalisation_type == "GN"):
        return GN_model()
    else:
        raise ValueError("Normlization has to be one of BN/LN/GN")



###### MODEL WITH BATCH NORMALISATION
class BN_model(nn.Module):
  def __init__(self):
    super(BN_model, self).__init__()
    ## INPUT BLOCK (BLOCK  1 )
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(dropout_value)   
    )# input - 28, output - 26
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(dropout_value)
    )#input - 26 , output - 26
    # TRANSITION BLOCK 
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )# input - 26 , output - 26
    self.pool1 = nn.MaxPool2d(2, 2)# input 26, output - 13
    # CONVOLUTION  BLOCK 2
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),#input = 13 , output  = 11
        nn.Dropout(dropout_value)
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),# input - 11 , output = 11
        nn.Dropout(dropout_value)
    )
    #   TRANSITION BLOCK 
    self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )
    # #   CONVOLUTION BLOCK 3
    self.conv7 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 8 , kernel_size = (3,3),bias = False),
        nn.BatchNorm2d(8),
        nn.ReLU(),#input = 11, output = 9
        nn.Dropout(dropout_value)
    )
    self.conv8 = nn.Sequential(
        nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),#input = 9, output = 7
        nn.Dropout(dropout_value)
    )
    self.conv9 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(dropout_value) 
    )#input - 7 , output - 5 
    #      CONVOLUTION BLOCK 4
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size = 5)
    )# output 1
    self.conv10 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = (1,1), bias = False)
    )


  def forward(self, x ):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.pool1(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    x = self.gap(x)
    x = self.conv10(x)
    x = x.view(-1,10)
    return F.log_softmax(x, dim = -1)

##### Model for LAYER NORMALISATION
class LN_model(nn.Module):
  def __init__(self):
    super(LN_model, self).__init__()
    ## INPUT BLOCK (BLOCK  1 )
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), bias = False),
        nn.ReLU(),
        nn.LayerNorm([8,26,26]),
        nn.Dropout(dropout_value)   
    )# input - 28, output - 26
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.ReLU(),
        nn.LayerNorm([16,26,26]),
        nn.Dropout(dropout_value)
    )#input - 26 , output - 26
    # TRANSITION BLOCK 
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )# input - 26 , output - 26
    self.pool1 = nn.MaxPool2d(2, 2)# input 26, output - 13
    # CONVOLUTION  BLOCK 2
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = (3,3), bias = False),
        nn.ReLU(),#input = 13 , output  = 11
        nn.LayerNorm([16,11,11]),
        nn.Dropout(dropout_value)
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.ReLU(),# input - 11 , output = 11
        nn.LayerNorm([16,11,11]),
        nn.Dropout(dropout_value)
    )
    #   TRANSITION BLOCK 
    self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )
    # #   CONVOLUTION BLOCK 3
    self.conv7 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 8 , kernel_size = (3,3),bias = False),
        nn.ReLU(),#input = 11, output = 9
        nn.LayerNorm([8,9,9]),
        nn.Dropout(dropout_value)
    )
    self.conv8 = nn.Sequential(
        nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.ReLU(),#input = 9, output = 7
        nn.LayerNorm([16,7,7]),
        nn.Dropout(dropout_value)
    )
    self.conv9 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.ReLU(),
        nn.LayerNorm([16,5,5]),
        nn.Dropout(dropout_value) 
    )#input - 7 , output - 5 
    #      CONVOLUTION BLOCK 4
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size = 5)
    )# output 1
    self.conv10 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = (1,1), bias = False)
    )


  def forward(self, x ):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.pool1(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    x = self.gap(x)
    x = self.conv10(x)
    x = x.view(-1,10)
    return F.log_softmax(x, dim = -1)


######  MODEL WITH GROUP NORMALISATION

class GN_model(nn.Module):
  def __init__(self):
    super(GN_model, self).__init__()
    ## INPUT BLOCK (BLOCK  1 )
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(dropout_value)   
    )# input - 28, output - 26
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(dropout_value)
    )#input - 26 , output - 26
    # TRANSITION BLOCK 
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )# input - 26 , output - 26
    self.pool1 = nn.MaxPool2d(2, 2)# input 26, output - 13
    # CONVOLUTION  BLOCK 2
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),#input = 13 , output  = 11
        nn.Dropout(dropout_value)
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),# input - 11 , output = 11
        nn.Dropout(dropout_value)
    )
    #   TRANSITION BLOCK 
    self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,1), bias = False)
    )
    # #   CONVOLUTION BLOCK 3
    self.conv7 = nn.Sequential(
        nn.Conv2d(in_channels = 8, out_channels = 8 , kernel_size = (3,3),bias = False),
        nn.BatchNorm2d(8),
        nn.ReLU(),#input = 11, output = 9
        nn.Dropout(dropout_value)
    )
    self.conv8 = nn.Sequential(
        nn.Conv2d(in_channels= 8, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),#input = 9, output = 7
        nn.Dropout(dropout_value)
    )
    self.conv9 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), bias = False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(dropout_value) 
    )#input - 7 , output - 5 
    #      CONVOLUTION BLOCK 4
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size = 5)
    )# output 1
    self.conv10 = nn.Sequential(
        nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = (1,1), bias = False)
    )


  def forward(self, x ):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.pool1(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    x = self.gap(x)
    x = self.conv10(x)
    x = x.view(-1,10)
    return F.log_softmax(x, dim = -1)


