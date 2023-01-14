# Solution of Session 3 Assignment
### PART 1

Excel Link: https://iitbacin-my.sharepoint.com/:x:/g/personal/210100140_iitb_ac_in/EUMFZbKeLTRJtGXzhgKzRdEBB8wX_2oz-dvpMI34tZe60w?e=crmFZA

Screenshots: 
<img src = "images/Screenshot 2023-01-14 at 4.02.23 AM.png">
<img src = "images/Screenshot 2023-01-14 at 4.02.43 AM.png">
<img src = "images/Screenshot 2023-01-14 at 4.02.53 AM.png">
<img src = "images/Screenshot 2023-01-14 at 4.05.42 AM.png">









### PART 2
#### Design of the neural network
I divided the full structure to 4 parts.To reduce the parameters i tried several methods and figured out the only way to do this is to reduce the number of channels and convolutions and also figured that keeping the channels low might now bring the accuracy that much down which take around 5oX the parameters as used in this case i.e. 16000.

Used ReLU, batchNorm2d, Dropout(0.1) after every layer as mentioned in the lecture except before the last layer.

#### 1st convolution block:

            nn.Conv2d(1 ,4, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(0.1),

            nn.Conv2d(4 ,8, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
            
            nn.MaxPool2d(2,2),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)

starting with image of 28*28, and channel size 1, 1st convolution coverts it to a channel size of 4 using 3 * 3 kernel but do not reduce the dimensions as padding is set to 1. Then used ReLU, BatchNorm2d, and Dropout at 0.1. 2nd convolution converts the channels to 8 but does not reduce the dimensions and rest things same. for the last part used MaxPool2d with stride 2 and kernel size 2 which reduces both x and y dimensions by half converting the image to 14*14 then applied and 1*1 convolution to merge the features of the channels and then as usual used ReLU, BatchNorm2d and Dropout.

#### 2nd Concolution block:
            nn.Conv2d(8 ,16, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.Conv2d(16 ,16, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.Conv2d(16 ,16, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
For the first convolution used the similar attributes as previous doubling the number of kernels from 8 to 16. Then added to 2 same convolutions outputing the same number of channels i.e. 16 and of the same dimension 14*!4. For the next convolution used MAxPool2d to reduce thedimension size by half to 7 * 7 then used a 1*1 convolutiont to combine the features and then as usual used ReLU, BatchNorm2d and Dropout.

#### 3rd Convolution Block
            nn.Conv2d(16 ,32, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32 ,16, 3 , padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),

            nn.Conv2d(16, 16, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
used 3 convolutions to first converting 16 channels to 32 channels without reducing the dimensions from 7*7 as that is the perfect dimension size before using GAP layer. Then used 2 convolutions to first convert 32 channels to 16 channels keeping the kernel size 3*3 and then used 1*1 kernel to merge the features from 16 channelsin 16 different channels.

#### 4th Convolution Block

        self.fc1 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 7)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 16, out_features = 10)
        )
first layer coverts the image dimension from 7*7*16 using global average pooling to change dimensions to 16,0,0 and then used a fully connected layer in which number of input features were 16 and output features were 10.

Parameteres Used: 16,490


Output logs running for 20 epochs:
  0%|          | 0/938 [00:00<?, ?it/s]<ipython-input-109-8c328c37ef63>:263: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.16144181787967682 batch_id=937: 100%|██████████| 938/938 [00:26<00:00, 35.91it/s]

Test set: Average loss: 0.1134, Accuracy: 9643/10000 (96%)

loss=0.08557406812906265 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.33it/s]

Test set: Average loss: 0.1116, Accuracy: 9644/10000 (96%)

loss=0.18140625953674316 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.54it/s]

Test set: Average loss: 0.0451, Accuracy: 9855/10000 (99%)

loss=0.04212013632059097 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.01it/s]

Test set: Average loss: 0.0385, Accuracy: 9876/10000 (99%)

loss=0.008542371913790703 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 40.79it/s]

Test set: Average loss: 0.0365, Accuracy: 9885/10000 (99%)

loss=0.013082927092909813 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 39.52it/s]

Test set: Average loss: 0.0383, Accuracy: 9878/10000 (99%)

loss=0.13993705809116364 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.14it/s]

Test set: Average loss: 0.0371, Accuracy: 9879/10000 (99%)

loss=0.044269781559705734 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.32it/s]

Test set: Average loss: 0.0269, Accuracy: 9921/10000 (99%)

loss=0.0303134024143219 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.65it/s]

Test set: Average loss: 0.0274, Accuracy: 9912/10000 (99%)

loss=0.12361900508403778 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 40.86it/s]

Test set: Average loss: 0.0279, Accuracy: 9921/10000 (99%)

loss=0.05279260873794556 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 40.84it/s]

Test set: Average loss: 0.0252, Accuracy: 9917/10000 (99%)

loss=0.054920706897974014 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 39.54it/s]

Test set: Average loss: 0.0276, Accuracy: 9917/10000 (99%)

loss=0.0541444756090641 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 40.89it/s]

Test set: Average loss: 0.0269, Accuracy: 9918/10000 (99%)

loss=0.16542279720306396 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.10it/s]

Test set: Average loss: 0.0307, Accuracy: 9899/10000 (99%)

loss=0.011109485290944576 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.34it/s]

Test set: Average loss: 0.0267, Accuracy: 9921/10000 (99%)

loss=0.004414476919919252 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 40.97it/s]

Test set: Average loss: 0.0221, Accuracy: 9930/10000 (99%)

loss=0.011795828118920326 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.04it/s]

Test set: Average loss: 0.0237, Accuracy: 9917/10000 (99%)

loss=0.08197896182537079 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.30it/s]

Test set: Average loss: 0.0315, Accuracy: 9895/10000 (99%)

loss=0.13735821843147278 batch_id=937: 100%|██████████| 938/938 [00:23<00:00, 40.06it/s]

Test set: Average loss: 0.0231, Accuracy: 9929/10000 (99%)

loss=0.2238200455904007 batch_id=937: 100%|██████████| 938/938 [00:22<00:00, 41.03it/s]

Test set: Average loss: 0.0180, Accuracy: 9947/10000 (99%)

####Touching the accuracy of 99.47%


