# TSAI - EVA8 Session 8 Assignment

## Problem Statement

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:  
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k] 
    2. Layer1 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]  
        2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        3. Add(X, R1)  
    3. Layer 2 -  
        1. Conv 3x3 [256k]  
        2. MaxPooling2D  
        3. BN  
        4. ReLU  
    4. Layer 3 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]  
        2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]  
        3. Add(X, R2)   
    5. MaxPooling with Kernel Size 4    
    6. FC Layer  
    7. SoftMax  
2. Uses One Cycle Policy such that:  
    1. Total Epochs = 24  
    2. Max at Epoch = 5  
    3. LRMIN = FIND  
    4. LRMAX = FIND  
    5. NO Annihilation  
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)  
4. Batch size = 512  
5. Target Accuracy: 90% (93.8% quadruple scores).  
6. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.  
7. Once done, proceed to answer the Assignment-Solution page. 

## Solution

This readme file provides a complete documentation for the custom ResNet architecture for CIFAR10 dataset. The custom ResNet architecture has been designed to meet the requirements specified in the problem statement.

## Architecture

The custom ResNet architecture is trained using One Cycle Policy with the following parameters:

Total epochs: 24
Maximum at epoch: 5
LRMIN: To be determined during training
LRMAX: To be determined during training
No Annihilation
The training data is transformed using the following operations:

RandomCrop (32, 32) after padding of 4
FlipLR
CutOut (8, 8)
Batch size: 512
The target accuracy for the custom ResNet architecture is 90% (93.8% quadruple scores).

## Modularity
The code for the custom ResNet architecture is modular and is available in the custom_resnet.py file in the GitHub repository. The collab is set up to import the custom_resnet package from GitHub and run the model.

## Conclusion
The custom ResNet architecture for CIFAR10 dataset has been designed and documented to meet the requirements specified in the problem statement. The code for the custom ResNet architecture is modular and is available in the GitHub repository. The target accuracy for the custom ResNet architecture is 90% (93.8% quadruple scores).


## Traning Logs
Batch_id=97 Loss=2.32335 Accuracy=30.77%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 1.6429, Accuracy: 4213/10000 (42.13%) EPOCH: 2 (LR: 0.006852006817213464)
Batch_id=97 Loss=1.46489 Accuracy=48.12%: 100%|██████████| 98/98 [00:30<00:00, 3.22it/s]
Test set: Average loss: 1.5617, Accuracy: 5072/10000 (50.72%) EPOCH: 3 (LR: 0.011324013634426928)
Batch_id=97 Loss=1.28735 Accuracy=56.73%: 100%|██████████| 98/98 [00:29<00:00, 3.28it/s]
Test set: Average loss: 1.1620, Accuracy: 6300/10000 (63.00%) EPOCH: 4 (LR: 0.015796020451640393)
Batch_id=97 Loss=1.09391 Accuracy=63.39%: 100%|██████████| 98/98 [00:30<00:00, 3.26it/s]
Test set: Average loss: 1.3301, Accuracy: 5818/10000 (58.18%) EPOCH: 5 (LR: 0.020268027268853857)
Batch_id=97 Loss=0.96296 Accuracy=67.22%: 100%|██████████| 98/98 [00:30<00:00, 3.27it/s]
Test set: Average loss: 0.9468, Accuracy: 6786/10000 (67.86%) EPOCH: 6 (LR: 0.023539955654761906)
Batch_id=97 Loss=0.82780 Accuracy=71.72%: 100%|██████████| 98/98 [00:30<00:00, 3.27it/s]
Test set: Average loss: 0.7929, Accuracy: 7327/10000 (73.27%) EPOCH: 7 (LR: 0.02230285148809524)
Batch_id=97 Loss=0.74122 Accuracy=74.94%: 100%|██████████| 98/98 [00:30<00:00, 3.24it/s]
Test set: Average loss: 0.7002, Accuracy: 7720/10000 (77.20%) EPOCH: 8 (LR: 0.021065747321428574)
Batch_id=97 Loss=0.66825 Accuracy=77.15%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 0.6524, Accuracy: 7806/10000 (78.06%) EPOCH: 9 (LR: 0.019828643154761905)
Batch_id=97 Loss=0.59791 Accuracy=79.63%: 100%|██████████| 98/98 [00:30<00:00, 3.26it/s]
Test set: Average loss: 0.5927, Accuracy: 8093/10000 (80.93%) EPOCH: 10 (LR: 0.01859153898809524)
Batch_id=97 Loss=0.57772 Accuracy=80.26%: 100%|██████████| 98/98 [00:29<00:00, 3.27it/s]
Test set: Average loss: 0.6061, Accuracy: 7941/10000 (79.41%) EPOCH: 11 (LR: 0.017354434821428573)
Batch_id=97 Loss=0.52784 Accuracy=81.98%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 0.5408, Accuracy: 8214/10000 (82.14%) EPOCH: 12 (LR: 0.016117330654761907)
Batch_id=97 Loss=0.49245 Accuracy=83.09%: 100%|██████████| 98/98 [00:30<00:00, 3.24it/s]
Test set: Average loss: 0.5216, Accuracy: 8303/10000 (83.03%) EPOCH: 13 (LR: 0.01488022648809524)
Batch_id=97 Loss=0.45059 Accuracy=84.36%: 100%|██████████| 98/98 [00:30<00:00, 3.24it/s]
Test set: Average loss: 0.4601, Accuracy: 8494/10000 (84.94%) EPOCH: 14 (LR: 0.013643122321428572)
Batch_id=97 Loss=0.44207 Accuracy=84.64%: 100%|██████████| 98/98 [00:30<00:00, 3.26it/s]
Test set: Average loss: 0.5304, Accuracy: 8276/10000 (82.76%) EPOCH: 15 (LR: 0.012406018154761906)
Batch_id=97 Loss=0.39831 Accuracy=86.25%: 100%|██████████| 98/98 [00:30<00:00, 3.21it/s]
Test set: Average loss: 0.4854, Accuracy: 8441/10000 (84.41%) EPOCH: 16 (LR: 0.011168913988095238)
Batch_id=97 Loss=0.38780 Accuracy=86.61%: 100%|██████████| 98/98 [00:29<00:00, 3.28it/s]
Test set: Average loss: 0.4659, Accuracy: 8517/10000 (85.17%) EPOCH: 17 (LR: 0.00993180982142857)
Batch_id=97 Loss=0.35138 Accuracy=87.70%: 100%|██████████| 98/98 [00:30<00:00, 3.24it/s]
Test set: Average loss: 0.4176, Accuracy: 8641/10000 (86.41%) EPOCH: 18 (LR: 0.008694705654761905)
Batch_id=97 Loss=0.32193 Accuracy=88.94%: 100%|██████████| 98/98 [00:29<00:00, 3.27it/s]
Test set: Average loss: 0.4320, Accuracy: 8599/10000 (85.99%) EPOCH: 19 (LR: 0.007457601488095239)
Batch_id=97 Loss=0.30037 Accuracy=89.61%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 0.3958, Accuracy: 8703/10000 (87.03%) EPOCH: 20 (LR: 0.00622049732142857)
Batch_id=97 Loss=0.27280 Accuracy=90.50%: 100%|██████████| 98/98 [00:30<00:00, 3.26it/s]
Test set: Average loss: 0.4053, Accuracy: 8686/10000 (86.86%) EPOCH: 21 (LR: 0.004983393154761904)
Batch_id=97 Loss=0.25960 Accuracy=91.06%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 0.3987, Accuracy: 8732/10000 (87.32%) EPOCH: 22 (LR: 0.003746288988095238)
Batch_id=97 Loss=0.23575 Accuracy=91.90%: 100%|██████████| 98/98 [00:30<00:00, 3.25it/s]
Test set: Average loss: 0.3703, Accuracy: 8823/10000 (88.23%) EPOCH: 23 (LR: 0.0025091848214285686)
Batch_id=97 Loss=0.22663 Accuracy=92.25%: 100%|██████████| 98/98 [00:30<00:00, 3.23it/s]
Test set: Average loss: 0.3615, Accuracy: 8848/10000 (88.48%) EPOCH: 24 (LR: 0.0012720806547619028)
Batch_id=97 Loss=0.20064 Accuracy=93.17%: 100%|██████████| 98/98 [00:30<00:00, 3.26it/s]
Test set: Average loss: 0.3494, Accuracy: 8888/10000 (88.88%)

## Accuracy achieved =  88.88%