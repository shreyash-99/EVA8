# Solution of Session 4 ASsignment
the answers in the assignments are not updated as i tried to attempt the question again and achieved the target which i wasnt able to achieve earlier.

## 1st Attempt out of 4

Target 
1. Get the set up and basic skeleton right and creating a light model 
2. Set tranforms 
3. Create dataloader, working code, training and teasting loop
4. applying all the necessary things such as batchnorm, dividing code in 4 blocks , using maxpooling etc.

Results
1. Parameters: 7.5k
2. Best Train Accuracy: 99.50
3. Best Test Accuracy: 98.95

Analysis
1. Model is overfitting, will try to reduce the gap between train and test accuracy using dropout.
2. Model is light, so for increasing the accuracies, we can make our model a little heavy by increasing the kernel sizes.
3. We can GAP layer also but will implement it later, and as implementing gap will reduce the parameters whcich can help us to increase channels also and make our model a little heavy


Model: 
https://colab.research.google.com/github/shreyash-99/EVA8/blob/main/Session_4_Assignment/Session%204%20Assignment%201%20of%204.ipynb#scrollTo=YFTMgmxua0Kr

Logs:
EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py:554: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Loss=0.0659426674246788 Batch_id=468 Accuracy=92.35: 100%|██████████| 469/469 [00:15<00:00, 30.10it/s]

Test set: Average loss: 0.0965, Accuracy: 9695/10000 (96.95%)

EPOCH: 1
Loss=0.052873145788908005 Batch_id=468 Accuracy=97.80: 100%|██████████| 469/469 [00:14<00:00, 31.54it/s]

Test set: Average loss: 0.0535, Accuracy: 9842/10000 (98.42%)

EPOCH: 2
Loss=0.05941098928451538 Batch_id=468 Accuracy=98.23: 100%|██████████| 469/469 [00:14<00:00, 31.63it/s]

Test set: Average loss: 0.0501, Accuracy: 9846/10000 (98.46%)

EPOCH: 3
Loss=0.005503728520125151 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:14<00:00, 31.71it/s]

Test set: Average loss: 0.0545, Accuracy: 9818/10000 (98.18%)

EPOCH: 4
Loss=0.07047614455223083 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:14<00:00, 31.80it/s]

Test set: Average loss: 0.0452, Accuracy: 9856/10000 (98.56%)

EPOCH: 5
Loss=0.018294747918844223 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:14<00:00, 31.61it/s]

Test set: Average loss: 0.0423, Accuracy: 9870/10000 (98.70%)

EPOCH: 6
Loss=0.021709101274609566 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:14<00:00, 31.37it/s]

Test set: Average loss: 0.0381, Accuracy: 9861/10000 (98.61%)

EPOCH: 7
Loss=0.04314659535884857 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:14<00:00, 31.61it/s]

Test set: Average loss: 0.0380, Accuracy: 9878/10000 (98.78%)

EPOCH: 8
Loss=0.003224683925509453 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:14<00:00, 31.75it/s]

Test set: Average loss: 0.0306, Accuracy: 9891/10000 (98.91%)

EPOCH: 9
Loss=0.006965493317693472 Batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:14<00:00, 31.82it/s]

Test set: Average loss: 0.0389, Accuracy: 9862/10000 (98.62%)

EPOCH: 10
Loss=0.017300236970186234 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:14<00:00, 31.65it/s]

Test set: Average loss: 0.0380, Accuracy: 9884/10000 (98.84%)

EPOCH: 11
Loss=0.006190266460180283 Batch_id=468 Accuracy=99.26: 100%|██████████| 469/469 [00:15<00:00, 30.83it/s]

Test set: Average loss: 0.0316, Accuracy: 9895/10000 (98.95%)

EPOCH: 12
Loss=0.046676065772771835 Batch_id=468 Accuracy=99.32: 100%|██████████| 469/469 [00:14<00:00, 31.59it/s]

Test set: Average loss: 0.0368, Accuracy: 9881/10000 (98.81%)

EPOCH: 13
Loss=0.03701046481728554 Batch_id=468 Accuracy=99.30: 100%|██████████| 469/469 [00:14<00:00, 31.73it/s]

Test set: Average loss: 0.0360, Accuracy: 9894/10000 (98.94%)

EPOCH: 14
Loss=0.011274625547230244 Batch_id=468 Accuracy=99.39: 100%|██████████| 469/469 [00:14<00:00, 31.94it/s]

Test set: Average loss: 0.0373, Accuracy: 9890/10000 (98.90%)

EPOCH: 15
Loss=0.0035534033086150885 Batch_id=468 Accuracy=99.39: 100%|██████████| 469/469 [00:14<00:00, 31.69it/s]

Test set: Average loss: 0.0338, Accuracy: 9886/10000 (98.86%)


## 2nd Attempt out of 4

Target
1. Reduce overfitting by implementing Dropout
2. Make the model a little heavy by modifing the channel sizes and skeleton

Results
1. Parameters: 10.2k
2. Best Train Accuracy: 98.71
3. Best Test Accuracy: 99.12

Analysis
1. The model is underfitting as we increased the diffficulty by using dropout of 0.1, our mission of decreasing the difference between test and train accuracy was achieved.
2. A little over the required parameters which can reduced by introducing global average pooling layer. because of which number of parameters will reduce whichb we compensate for putting more layers.
3. In the next model, will try to decrease the dropout value to 0.05 and add image augmentation.

Model: https://colab.research.google.com/github/shreyash-99/EVA8/blob/main/Session_4_Assignment/Session%204%20Assignment%202%20of%204.ipynb#scrollTo=aOAr4P71hx6X

Logs: 
EPOCH: 0
Loss=0.35013487935066223 Batch_id=468 Accuracy=89.11: 100%|██████████| 469/469 [00:15<00:00, 31.05it/s]

Test set: Average loss: 0.1117, Accuracy: 9647/10000 (96.47%)

EPOCH: 1
Loss=0.09669745713472366 Batch_id=468 Accuracy=96.36: 100%|██████████| 469/469 [00:15<00:00, 30.81it/s]

Test set: Average loss: 0.0700, Accuracy: 9772/10000 (97.72%)

EPOCH: 2
Loss=0.0865863785147667 Batch_id=468 Accuracy=97.23: 100%|██████████| 469/469 [00:16<00:00, 29.19it/s]

Test set: Average loss: 0.0530, Accuracy: 9827/10000 (98.27%)

EPOCH: 3
Loss=0.03444383665919304 Batch_id=468 Accuracy=97.58: 100%|██████████| 469/469 [00:16<00:00, 28.92it/s]

Test set: Average loss: 0.0423, Accuracy: 9868/10000 (98.68%)

EPOCH: 4
Loss=0.08114390075206757 Batch_id=468 Accuracy=97.77: 100%|██████████| 469/469 [00:15<00:00, 30.89it/s]

Test set: Average loss: 0.0461, Accuracy: 9856/10000 (98.56%)

EPOCH: 5
Loss=0.12626083195209503 Batch_id=468 Accuracy=98.00: 100%|██████████| 469/469 [00:15<00:00, 30.87it/s]

Test set: Average loss: 0.0374, Accuracy: 9875/10000 (98.75%)

EPOCH: 6
Loss=0.06966187804937363 Batch_id=468 Accuracy=98.04: 100%|██████████| 469/469 [00:15<00:00, 30.54it/s]

Test set: Average loss: 0.0355, Accuracy: 9883/10000 (98.83%)

EPOCH: 7
Loss=0.03103007562458515 Batch_id=468 Accuracy=98.19: 100%|██████████| 469/469 [00:15<00:00, 30.56it/s]

Test set: Average loss: 0.0490, Accuracy: 9854/10000 (98.54%)

EPOCH: 8
Loss=0.11881162971258163 Batch_id=468 Accuracy=98.17: 100%|██████████| 469/469 [00:14<00:00, 31.54it/s]

Test set: Average loss: 0.0381, Accuracy: 9888/10000 (98.88%)

EPOCH: 9
Loss=0.06233637407422066 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:15<00:00, 31.08it/s]

Test set: Average loss: 0.0337, Accuracy: 9894/10000 (98.94%)

EPOCH: 10
Loss=0.07006912678480148 Batch_id=468 Accuracy=98.41: 100%|██████████| 469/469 [00:15<00:00, 30.86it/s]

Test set: Average loss: 0.0289, Accuracy: 9908/10000 (99.08%)

EPOCH: 11
Loss=0.015529781579971313 Batch_id=468 Accuracy=98.39: 100%|██████████| 469/469 [00:15<00:00, 30.79it/s]

Test set: Average loss: 0.0373, Accuracy: 9878/10000 (98.78%)

EPOCH: 12
Loss=0.055175915360450745 Batch_id=468 Accuracy=98.49: 100%|██████████| 469/469 [00:16<00:00, 29.18it/s]

Test set: Average loss: 0.0412, Accuracy: 9869/10000 (98.69%)

EPOCH: 13
Loss=0.013990357518196106 Batch_id=468 Accuracy=98.54: 100%|██████████| 469/469 [00:15<00:00, 30.65it/s]

Test set: Average loss: 0.0309, Accuracy: 9905/10000 (99.05%)

EPOCH: 14
Loss=0.03938385844230652 Batch_id=468 Accuracy=98.63: 100%|██████████| 469/469 [00:14<00:00, 31.37it/s]

Test set: Average loss: 0.0265, Accuracy: 9909/10000 (99.09%)

EPOCH: 15
Loss=0.05305905640125275 Batch_id=468 Accuracy=98.63: 100%|██████████| 469/469 [00:14<00:00, 31.41it/s]

Test set: Average loss: 0.0293, Accuracy: 9903/10000 (99.03%)

## 3rd Attempt out of 4

Target
1. Will be adding GAP layer and increase the heaviness of the model
2. Apply augmentation techniques
4. Tried to make bias false during the convolutions
5. Also i noticed keeping 2 max pools ir reducing the accuracies and it cant reach even 99.2 consistently therfore will to reduce it and add more convolution layer
6. Also noticed that you didnt add BatchNorm and dropout after the convoulution with kernel size 1, will try to do it and see results
7. Also will try to reduce the number of convolution blocks(as in the lecture) and see the results.

Results:
1. Parameters: 10.5k
2. Best Test Accuracy: 99.45(in 15 epochs)
3. Best Train Accuracy: 98.96

Conclusion:
1. Model is good, no overfitting and achieved around the target percentage in last 3 of 15 epochs without even implementing StepLR
2. Added GAP layer and increased the convolution layers in the end
3. Making bias False didnt make much of a difference
4. Removing the 2nd Maxpool and adding some convolution layers in that place increased both the accuracies
5. Also tried adding batchnorm and dropout for the transition layers whihch made the model worse so removed it.
6. Tried to reduce the layers as in lecture there were just 6 and increasing the channel size but it didnt gave results therefor added more layers and decreased the channel size a bit which proved to be a bit better.
7. AT LAST, i will try StepLR next and will try to reduce some parametres to bring it under 10k, which i know should have been done earlier.

Model: https://colab.research.google.com/github/shreyash-99/EVA8/blob/main/Session_4_Assignment/Session%204%20Assignment%203%20of%204.ipynb#scrollTo=z9rDLXaLZLLo

Logs: 
EPOCH: 0
Loss=0.22045810520648956 Batch_id=468 Accuracy=81.80: 100%|██████████| 469/469 [00:24<00:00, 19.52it/s]

Test set: Average loss: 0.0957, Accuracy: 9721/10000 (97.21%)

EPOCH: 1
Loss=0.03583424538373947 Batch_id=468 Accuracy=97.11: 100%|██████████| 469/469 [00:23<00:00, 20.29it/s]

Test set: Average loss: 0.0482, Accuracy: 9869/10000 (98.69%)

EPOCH: 2
Loss=0.052883077412843704 Batch_id=468 Accuracy=97.86: 100%|██████████| 469/469 [00:19<00:00, 24.06it/s]

Test set: Average loss: 0.0379, Accuracy: 9874/10000 (98.74%)

EPOCH: 3
Loss=0.028870798647403717 Batch_id=468 Accuracy=98.14: 100%|██████████| 469/469 [00:20<00:00, 22.46it/s]

Test set: Average loss: 0.0351, Accuracy: 9883/10000 (98.83%)

EPOCH: 4
Loss=0.0675908550620079 Batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:19<00:00, 23.95it/s]

Test set: Average loss: 0.0350, Accuracy: 9886/10000 (98.86%)

EPOCH: 5
Loss=0.03955832123756409 Batch_id=468 Accuracy=98.43: 100%|██████████| 469/469 [00:20<00:00, 23.24it/s]

Test set: Average loss: 0.0308, Accuracy: 9904/10000 (99.04%)

EPOCH: 6
Loss=0.048263002187013626 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:20<00:00, 23.36it/s]

Test set: Average loss: 0.0228, Accuracy: 9930/10000 (99.30%)

EPOCH: 7
Loss=0.013590446673333645 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:19<00:00, 23.75it/s]

Test set: Average loss: 0.0228, Accuracy: 9925/10000 (99.25%)

EPOCH: 8
Loss=0.029483161866664886 Batch_id=468 Accuracy=98.74: 100%|██████████| 469/469 [00:20<00:00, 23.04it/s]

Test set: Average loss: 0.0235, Accuracy: 9923/10000 (99.23%)

EPOCH: 9
Loss=0.01748531498014927 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:20<00:00, 23.07it/s]

Test set: Average loss: 0.0222, Accuracy: 9938/10000 (99.38%)

EPOCH: 10
Loss=0.0527176596224308 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:21<00:00, 21.52it/s]
Test set: Average loss: 0.0250, Accuracy: 9915/10000 (99.15%)

EPOCH: 11
Loss=0.03224899247288704 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:20<00:00, 22.85it/s]

Test set: Average loss: 0.0228, Accuracy: 9929/10000 (99.29%)

EPOCH: 12
Loss=0.07122963666915894 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:19<00:00, 23.92it/s]

Test set: Average loss: 0.0231, Accuracy: 9924/10000 (99.24%)

EPOCH: 13
Loss=0.06272076070308685 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:19<00:00, 24.02it/s]

Test set: Average loss: 0.0178, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.028698936104774475 Batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:19<00:00, 23.77it/s]

Test set: Average loss: 0.0189, Accuracy: 9945/10000 (99.45%)

EPOCH: 15
Loss=0.026847153902053833 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:19<00:00, 23.84it/s]

Test set: Average loss: 0.0191, Accuracy: 9941/10000 (99.41%)


## 4th Attempt out of 4

Target 
1. Firstly implement stepLR to get consistent results
2. Reduce the size by a little bit to make it under 10k by changing some channel sizes
3. Will also try to increase to initial lr.

Results:
1. Parameters: 9.3k
2. Best Test Accuracy: 99.50% (till 15th epoch)
3. Best Train Accuracy: 99.21% (till 15th epoch)

Conclusion:
1. Model is very good as it is not at all overfitting, results are good it achieves 99.4% accuracy from 7th epoch 
2. Dont know how reducing the but reducing the channel the size of channels(16->8) for first was better for the model.
3. Increasing the lr to 0.3 also helped to achieve the target fast and steplr helped it to reduced it so that the target doesnt overshoot.

Model : https://colab.research.google.com/drive/1HPWIZeY2q3r9J-y_yTlalb7TMpvBxQm_?usp=sharing

Logs: 
EPOCH: 0 Learning Rate:  [0.03]
Loss=0.13147200644016266 Batch_id=468 Accuracy=89.32: 100%|██████████| 469/469 [00:18<00:00, 25.69it/s]

Test set: Average loss: 0.0672, Accuracy: 9784/10000 (97.84%)

EPOCH: 1 Learning Rate:  [0.03]
Loss=0.026439359411597252 Batch_id=468 Accuracy=97.81: 100%|██████████| 469/469 [00:17<00:00, 26.10it/s]

Test set: Average loss: 0.0539, Accuracy: 9836/10000 (98.36%)

EPOCH: 2 Learning Rate:  [0.03]
Loss=0.10534457117319107 Batch_id=468 Accuracy=98.23: 100%|██████████| 469/469 [00:18<00:00, 25.39it/s]

Test set: Average loss: 0.0330, Accuracy: 9894/10000 (98.94%)

EPOCH: 3 Learning Rate:  [0.03]
Loss=0.03889813646674156 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:18<00:00, 25.96it/s]

Test set: Average loss: 0.0258, Accuracy: 9913/10000 (99.13%)

EPOCH: 4 Learning Rate:  [0.03]
Loss=0.18354378640651703 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:18<00:00, 25.46it/s]

Test set: Average loss: 0.0323, Accuracy: 9893/10000 (98.93%)

EPOCH: 5 Learning Rate:  [0.03]
Loss=0.023572811856865883 Batch_id=468 Accuracy=98.69: 100%|██████████| 469/469 [00:19<00:00, 24.57it/s]

Test set: Average loss: 0.0316, Accuracy: 9906/10000 (99.06%)

EPOCH: 6 Learning Rate:  [0.003]
Loss=0.017485691234469414 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:17<00:00, 26.34it/s]

Test set: Average loss: 0.0225, Accuracy: 9930/10000 (99.30%)

EPOCH: 7 Learning Rate:  [0.003]
Loss=0.013596788048744202 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:17<00:00, 26.21it/s]

Test set: Average loss: 0.0203, Accuracy: 9945/10000 (99.45%)

EPOCH: 8 Learning Rate:  [0.003]
Loss=0.07576064765453339 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:18<00:00, 25.99it/s]

Test set: Average loss: 0.0200, Accuracy: 9947/10000 (99.47%)

EPOCH: 9 Learning Rate:  [0.003]
Loss=0.09534329921007156 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:17<00:00, 26.43it/s]

Test set: Average loss: 0.0195, Accuracy: 9945/10000 (99.45%)

EPOCH: 10 Learning Rate:  [0.003]
Loss=0.0043024662882089615 Batch_id=468 Accuracy=99.07: 100%|██████████| 469/469 [00:17<00:00, 27.07it/s]

Test set: Average loss: 0.0183, Accuracy: 9950/10000 (99.50%)

EPOCH: 11 Learning Rate:  [0.003]
Loss=0.0062692477367818356 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:17<00:00, 26.54it/s]

Test set: Average loss: 0.0188, Accuracy: 9948/10000 (99.48%)

EPOCH: 12 Learning Rate:  [0.00030000000000000003]
Loss=0.01717514358460903 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:17<00:00, 26.85it/s]

Test set: Average loss: 0.0189, Accuracy: 9948/10000 (99.48%)

EPOCH: 13 Learning Rate:  [0.00030000000000000003]
Loss=0.002918594516813755 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:17<00:00, 27.02it/s]

Test set: Average loss: 0.0192, Accuracy: 9948/10000 (99.48%)

EPOCH: 14 Learning Rate:  [0.00030000000000000003]
Loss=0.006048992276191711 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:18<00:00, 25.06it/s]

Test set: Average loss: 0.0191, Accuracy: 9949/10000 (99.49%)

EPOCH: 15 Learning Rate:  [0.00030000000000000003]
Loss=0.0249591376632452 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:17<00:00, 26.47it/s]

Test set: Average loss: 0.0188, Accuracy: 9948/10000 (99.48%)

EPOCH: 16 Learning Rate:  [0.00030000000000000003]
Loss=0.033999521285295486 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:17<00:00, 26.42it/s]

Test set: Average loss: 0.0189, Accuracy: 9948/10000 (99.48%)

EPOCH: 17 Learning Rate:  [0.00030000000000000003]
Loss=0.03355110064148903 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:18<00:00, 25.98it/s]

Test set: Average loss: 0.0184, Accuracy: 9949/10000 (99.49%)

EPOCH: 18 Learning Rate:  [3.0000000000000004e-05]
Loss=0.11394136399030685 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:18<00:00, 25.81it/s]

Test set: Average loss: 0.0184, Accuracy: 9947/10000 (99.47%)

EPOCH: 19 Learning Rate:  [3.0000000000000004e-05]
Loss=0.0030106117483228445 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:17<00:00, 26.44it/s]

Test set: Average loss: 0.0182, Accuracy: 9947/10000 (99.47%)
