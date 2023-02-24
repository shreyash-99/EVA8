import torch.nn as nn
import torch.nn.functional as F

#depthwise separable convolution
class BasicBlock(nn.Module):
    expansion = 1
  
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.in_chan = in_ch
        self.out_chan = out_ch

        self.conv1 = nn.Conv2d(self.in_chan, self.out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_chan)
        self.conv2 = nn.Conv2d(self.out_chan, self.out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_chan)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.in_chan != self.expansion*self.out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_chan, self.expansion*self.out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*self.out_chan)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class custom_ResNet(nn.Module):
    def __init__(self):
        super(custom_ResNet, self).__init__()
        ### input = 32*32 , output = 32*32 , out channels = 32
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #### input = 32* 32, output = 16 * 16
        self.Layer1_part1 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, stride = 1, padding = 1, kernel_size = 3, bias = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Layer1_part2 = nn.Sequential(
            BasicBlock(64, 128, 2)
        )

        ##### input = 16 * 16 output = 8 * 8
        self.Layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )   

        #### input = 8 * 8 output = 4 * 4
        self.Layer3_part1 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.MaxPool2d(kernel_size = 2, stride =2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) 
        self.Layer3_part2 = nn.Sequential(
            BasicBlock(256,512,2)
        )

        #### input = 4 * 4 * 512 --> 1 * 1 * 512 --> 512 --> 10
        self.maxpool = nn.MaxPool2d(kernel_size = 4)
        self.fc = nn.Linear(in_features = 512, out_features = 10, bias = False)


    def forward(self, x):
        x = self.prep(x)
        x1 = self.Layer1_part1(x)
        x2 = self.Layer1_part2(x)
        x = x1 + x2
        x = self.Layer2(x)
        x3 = self.Layer3_part1(x)
        x4 = self.Layer3_part2(x)
        x = x4 + x3
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x