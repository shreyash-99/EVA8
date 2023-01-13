# EVA8 - Assignment 2.5

### Data Representation
Firstly installed the MNIST dataset and transformed it to tensors<br />
Created my own dataset named myData and made it iterable by adding __getitem__ and __len__ function.<br/>
Each element of myData contained 4 parts 1st one was image, 2nd was label of the image, 3th was the random number(in hot encoded form), 4th was the label 
for sum(which was image label + random number)<br/>
image - [1x28x28] <br/>
image_label- int<br/>
random_num - [10] eg 6 = [0,0,0,0,0,0,1,0,0,0]<br/>
final_label - int<br/>

### Random Data Generation
Installed the random library and used random.rand(0,9) to create a random number from [0,9]<br/>
Now main task was to convert to a form which could be send to neural network.So i thought we should enter this data inside networka as full connected layer
, therefor converted it to a 2D tensor using indirect way of conveting to hot encoder.

## Neural Network
used the strategy for increasing and decreasing the channels for convolution neural network.<br/>
I combined the 2 datatsets at last and first predictd the mnist image till end(i.e. output is fully connected layer of 10 features)<br/>
Combined the datasets by concatenating along the columns as both of them were of size [batch_size, 10] which made a single fully connected layer of 
dimension [batch_size,20]<br/>
Rest Information about layers can be found in the comments provided in the code

> In between followed all the procedures taught in lecture like initially checking the data by visualising it, sending an image first to neural network 
then sending a single batch of image and then only sending 1 epoch to network etc. (can be seen in the code)

## Training and evaluation of Network
##### For CUDA
use_cuda = torch.cuda.is_available()<br/>
device = torch.device("cuda" if use_cuda else "cpu")<br/>
sent the instance of MyNetwork to gpu by writing network = MyNetwotk.to(device)<br/>
also when sending the images and random_num data, first store these in gpu and then send to the network
##### Optimiser 
Used optim.Adam as this is better than optim.SGD and others when we are with 2 outputs
#### Losses
captured 2 losses, loss1 and loss2 differently for both mnist data and sum prediction<br/>
used F.cross.entropy(...) for calculating losses as this better than nll_loss and works for both loss1 case and loss2 case.
#### Backpropagation
loss = loss1 + loss2<br/>
then backpropagated loss and updated the gradients of weights
#### Evaluation of results
evaluated total number of correct prediction in each epoch using function get_num_correct(preds, labels)
>def get_num_correct(preds, labels):<br/>
    return preds.argmax(dim = 1).eq(labels).sum().item()<br/>


## Results

epoch: 0 total correct for image : 45230 total correct for sum : 8101 loss for image: 1024.3916670084 loss for sum: 1754.7015159130096<br/>
epoch: 1 total correct for image : 56226 total correct for sum : 13508 loss for image: 914.4627431631088 loss for sum: 1733.7743835449219<br/>
epoch: 2 total correct for image : 57177 total correct for sum : 17458 loss for image: 904.9028449058533 loss for sum: 1712.4266505241394<br/>
epoch: 3 total correct for image : 57438 total correct for sum : 19994 loss for image: 902.2632160186768 loss for sum: 1686.1123733520508<br/>
epoch: 4 total correct for image : 57423 total correct for sum : 22899 loss for image: 902.3710641860962 loss for sum: 1657.1524307727814<br/>
epoch: 5 total correct for image : 57631 total correct for sum : 26364 loss for image: 900.2642750740051 loss for sum: 1627.9684963226318<br/>
epoch: 6 total correct for image : 57666 total correct for sum : 28086 loss for image: 900.0357085466385 loss for sum: 1601.3217034339905<br/>
epoch: 7 total correct for image : 57624 total correct for sum : 28658 loss for image: 900.3416709899902 loss for sum: 1582.9965052604675<br/>
epoch: 8 total correct for image : 57823 total correct for sum : 28810 loss for image: 898.3947875499725 loss for sum: 1569.271897315979<br/>
epoch: 9 total correct for image : 57447 total correct for sum : 29606 loss for image: 902.1912475824356 loss for sum: 1557.1763849258423<br/>

Hitting 96% accuracy for mnist data and approx 50% accuracy for sum
