# TSAI - EVA8 Session 10 Assignment

## Problem Statement

Check out this [network](https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/vit.py):

1. Re-write this network such that it is similar to the network we wrote in the class  
2. All parameters are the same as the network we wrote  
3. Proceed to submit the assignment:  
    1. Share the model code and link to the model cost  
    2. Share the training logs  
    3. Share the gradcam images for 10 misclassified images. 


## Solution

### Model
Modified model, [link](./vit_modified.py) <br>
Implementation, [Python Notebook](./Session_10_Assignment.ipynb)<br>

## What I implemented
1. Modified the model to use nn.Conv instead of nn.Linear whihc involved changing the dimension between batch_size, patch, height, width again and again as it will be very different when using nn.Conv and when using nn.Linear and einops made this rearraging very easy.<br>
2. Also now when classifying, the model now uses the cls token(the first embedding) to make prediction rather than using all of them.

## Model Summary
        Layer (type)               Output Shape         Param   <br>
            Conv2d-1            [-1, 128, 8, 8]           6,272<br>
         Rearrange-2              [-1, 64, 128]               0<br>
           Dropout-3              [-1, 65, 128]               0<br>
         LayerNorm-4              [-1, 65, 128]             256<br>
         Rearrange-5                [-1, 2, 65]               0<br>
            Conv2d-6                [-1, 2, 65]         198,144<br>
         Rearrange-7             [-1, 65, 1536]               0<br>
           Softmax-8            [-1, 8, 65, 65]               0<br>
         Rearrange-9                [-1, 2, 65]               0<br>
           Conv2d-10                [-1, 2, 65]          65,664<br>
        Rearrange-11              [-1, 65, 128]               0<br>
          Dropout-12              [-1, 65, 128]               0<br>
        Attention-13              [-1, 65, 128]               0<br>
          PreNorm-14              [-1, 65, 128]               0<br>
        LayerNorm-15              [-1, 65, 128]             256<br>
           Conv2d-16                [-1, 65, 2]          33,024<br>
             GELU-17                [-1, 65, 2]               0<br>
          Dropout-18                [-1, 65, 2]               0<br>
           Conv2d-19                [-1, 65, 2]          32,896<br>
          Dropout-20                [-1, 65, 2]               0<br>
      FeedForward-21              [-1, 65, 128]               0<br>
          PreNorm-22              [-1, 65, 128]               0<br>
        LayerNorm-23              [-1, 65, 128]             256<br>
        Rearrange-24                [-1, 2, 65]               0<br>
           Conv2d-25                [-1, 2, 65]         198,144<br>
        Rearrange-26             [-1, 65, 1536]               0<br>
          Softmax-27            [-1, 8, 65, 65]               0<br>
        Rearrange-28                [-1, 2, 65]               0<br>
           Conv2d-29                [-1, 2, 65]          65,664<br>
        Rearrange-30              [-1, 65, 128]               0<br>
          Dropout-31              [-1, 65, 128]               0<br>
        Attention-32              [-1, 65, 128]               0<br>
          PreNorm-33              [-1, 65, 128]               0<br>
        LayerNorm-34              [-1, 65, 128]             256<br>
           Conv2d-35                [-1, 65, 2]          33,024<br>
             GELU-36                [-1, 65, 2]               0<br>
          Dropout-37                [-1, 65, 2]               0<br>
           Conv2d-38                [-1, 65, 2]          32,896<br>
          Dropout-39                [-1, 65, 2]               0<br>
      FeedForward-40              [-1, 65, 128]               0<br>
          PreNorm-41              [-1, 65, 128]               0<br>
        LayerNorm-42              [-1, 65, 128]             256<br>
        Rearrange-43                [-1, 2, 65]               0<br>
           Conv2d-44                [-1, 2, 65]         198,144<br>
        Rearrange-45             [-1, 65, 1536]               0<br>
          Softmax-46            [-1, 8, 65, 65]               0<br>
        Rearrange-47                [-1, 2, 65]               0<br>
           Conv2d-48                [-1, 2, 65]          65,664<br>
        Rearrange-49              [-1, 65, 128]               0<br>
          Dropout-50              [-1, 65, 128]               0<br>
        Attention-51              [-1, 65, 128]               0<br>
          PreNorm-52              [-1, 65, 128]               0<br>
        LayerNorm-53              [-1, 65, 128]             256<br>
           Conv2d-54                [-1, 65, 2]          33,024<br>
             GELU-55                [-1, 65, 2]               0<br>
          Dropout-56                [-1, 65, 2]               0<br>
           Conv2d-57                [-1, 65, 2]          32,896<br>
          Dropout-58                [-1, 65, 2]               0<br>
      FeedForward-59              [-1, 65, 128]               0<br>
          PreNorm-60              [-1, 65, 128]               0<br>
        LayerNorm-61              [-1, 65, 128]             256<br>
        Rearrange-62                [-1, 2, 65]               0<br>
           Conv2d-63                [-1, 2, 65]         198,144<br>
        Rearrange-64             [-1, 65, 1536]               0<br>
          Softmax-65            [-1, 8, 65, 65]               0<br>
        Rearrange-66                [-1, 2, 65]               0<br>
           Conv2d-67                [-1, 2, 65]          65,664<br>
        Rearrange-68              [-1, 65, 128]               0<br>
          Dropout-69              [-1, 65, 128]               0<br>
        Attention-70              [-1, 65, 128]               0<br>
          PreNorm-71              [-1, 65, 128]               0<br>
        LayerNorm-72              [-1, 65, 128]             256<br>
           Conv2d-73                [-1, 65, 2]          33,024<br>
             GELU-74                [-1, 65, 2]               0<br>
          Dropout-75                [-1, 65, 2]               0<br>
           Conv2d-76                [-1, 65, 2]          32,896<br>
          Dropout-77                [-1, 65, 2]               0<br>
      FeedForward-78              [-1, 65, 128]               0<br>
          PreNorm-79              [-1, 65, 128]               0<br>
        LayerNorm-80              [-1, 65, 128]             256<br>
        Rearrange-81                [-1, 2, 65]               0<br>
           Conv2d-82                [-1, 2, 65]         198,144<br>
        Rearrange-83             [-1, 65, 1536]               0<br>
          Softmax-84            [-1, 8, 65, 65]               0<br>
        Rearrange-85                [-1, 2, 65]               0<br>
           Conv2d-86                [-1, 2, 65]          65,664<br>
        Rearrange-87              [-1, 65, 128]               0<br>
          Dropout-88              [-1, 65, 128]               0<br>
        Attention-89              [-1, 65, 128]               0<br>
          PreNorm-90              [-1, 65, 128]               0<br>
        LayerNorm-91              [-1, 65, 128]             256<br>
           Conv2d-92                [-1, 65, 2]          33,024<br>
             GELU-93                [-1, 65, 2]               0<br>
          Dropout-94                [-1, 65, 2]               0<br>
           Conv2d-95                [-1, 65, 2]          32,896<br>
          Dropout-96                [-1, 65, 2]               0<br>
      FeedForward-97              [-1, 65, 128]               0<br>
          PreNorm-98              [-1, 65, 128]               0<br>
        LayerNorm-99              [-1, 65, 128]             256<br>
       Rearrange-100                [-1, 2, 65]               0<br>
          Conv2d-101                [-1, 2, 65]         198,144<br>
       Rearrange-102             [-1, 65, 1536]               0<br>
         Softmax-103            [-1, 8, 65, 65]               0<br>
       Rearrange-104                [-1, 2, 65]               0<br>
          Conv2d-105                [-1, 2, 65]          65,664<br>
       Rearrange-106              [-1, 65, 128]               0<br>
         Dropout-107              [-1, 65, 128]               0<br>
       Attention-108              [-1, 65, 128]               0<br>
         PreNorm-109              [-1, 65, 128]               0<br>
       LayerNorm-110              [-1, 65, 128]             256<br>
          Conv2d-111                [-1, 65, 2]          33,024<br>
            GELU-112                [-1, 65, 2]               0<br>
         Dropout-113                [-1, 65, 2]               0<br>
          Conv2d-114                [-1, 65, 2]          32,896<br>
         Dropout-115                [-1, 65, 2]               0<br>
     FeedForward-116              [-1, 65, 128]               0<br>
         PreNorm-117              [-1, 65, 128]               0<br>
     Transformer-118              [-1, 65, 128]               0<br>
        Identity-119                  [-1, 128]               0<br>
       LayerNorm-120                  [-1, 128]             256<br>
       Rearrange-121                 [-1, 2, 1]               0<br>
          Conv2d-122                 [-1, 2, 1]           1,290<br>
       Rearrange-123                   [-1, 10]               0<br>
Total params: 1,989,258
Trainable params: 1,989,258
Non-trainable params: 0
<br>
Input size (MB): 0.01
Forward/backward pass size (MB): 9.47
Params size (MB): 7.59
Estimated Total Size (MB): 17.07
<br>
### Training Logs

![training logs](./images/training_logs.png)

