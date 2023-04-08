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
Modified model, [link](/vit_modified.py) 

## Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 128, 8, 8]           6,272
         Rearrange-2              [-1, 64, 128]               0
           Dropout-3              [-1, 65, 128]               0
         LayerNorm-4              [-1, 65, 128]             256
         Rearrange-5                [-1, 2, 65]               0
            Conv2d-6                [-1, 2, 65]         198,144
         Rearrange-7             [-1, 65, 1536]               0
           Softmax-8            [-1, 8, 65, 65]               0
         Rearrange-9                [-1, 2, 65]               0
           Conv2d-10                [-1, 2, 65]          65,664
        Rearrange-11              [-1, 65, 128]               0
          Dropout-12              [-1, 65, 128]               0
        Attention-13              [-1, 65, 128]               0
          PreNorm-14              [-1, 65, 128]               0
        LayerNorm-15              [-1, 65, 128]             256
           Conv2d-16                [-1, 65, 2]          33,024
             GELU-17                [-1, 65, 2]               0
          Dropout-18                [-1, 65, 2]               0
           Conv2d-19                [-1, 65, 2]          32,896
          Dropout-20                [-1, 65, 2]               0
      FeedForward-21              [-1, 65, 128]               0
          PreNorm-22              [-1, 65, 128]               0
        LayerNorm-23              [-1, 65, 128]             256
        Rearrange-24                [-1, 2, 65]               0
           Conv2d-25                [-1, 2, 65]         198,144
        Rearrange-26             [-1, 65, 1536]               0
          Softmax-27            [-1, 8, 65, 65]               0
        Rearrange-28                [-1, 2, 65]               0
           Conv2d-29                [-1, 2, 65]          65,664
        Rearrange-30              [-1, 65, 128]               0
          Dropout-31              [-1, 65, 128]               0
        Attention-32              [-1, 65, 128]               0
          PreNorm-33              [-1, 65, 128]               0
        LayerNorm-34              [-1, 65, 128]             256
           Conv2d-35                [-1, 65, 2]          33,024
             GELU-36                [-1, 65, 2]               0
          Dropout-37                [-1, 65, 2]               0
           Conv2d-38                [-1, 65, 2]          32,896
          Dropout-39                [-1, 65, 2]               0
      FeedForward-40              [-1, 65, 128]               0
          PreNorm-41              [-1, 65, 128]               0
        LayerNorm-42              [-1, 65, 128]             256
        Rearrange-43                [-1, 2, 65]               0
           Conv2d-44                [-1, 2, 65]         198,144
        Rearrange-45             [-1, 65, 1536]               0
          Softmax-46            [-1, 8, 65, 65]               0
        Rearrange-47                [-1, 2, 65]               0
           Conv2d-48                [-1, 2, 65]          65,664
        Rearrange-49              [-1, 65, 128]               0
          Dropout-50              [-1, 65, 128]               0
        Attention-51              [-1, 65, 128]               0
          PreNorm-52              [-1, 65, 128]               0
        LayerNorm-53              [-1, 65, 128]             256
           Conv2d-54                [-1, 65, 2]          33,024
             GELU-55                [-1, 65, 2]               0
          Dropout-56                [-1, 65, 2]               0
           Conv2d-57                [-1, 65, 2]          32,896
          Dropout-58                [-1, 65, 2]               0
      FeedForward-59              [-1, 65, 128]               0
          PreNorm-60              [-1, 65, 128]               0
        LayerNorm-61              [-1, 65, 128]             256
        Rearrange-62                [-1, 2, 65]               0
           Conv2d-63                [-1, 2, 65]         198,144
        Rearrange-64             [-1, 65, 1536]               0
          Softmax-65            [-1, 8, 65, 65]               0
        Rearrange-66                [-1, 2, 65]               0
           Conv2d-67                [-1, 2, 65]          65,664
        Rearrange-68              [-1, 65, 128]               0
          Dropout-69              [-1, 65, 128]               0
        Attention-70              [-1, 65, 128]               0
          PreNorm-71              [-1, 65, 128]               0
        LayerNorm-72              [-1, 65, 128]             256
           Conv2d-73                [-1, 65, 2]          33,024
             GELU-74                [-1, 65, 2]               0
          Dropout-75                [-1, 65, 2]               0
           Conv2d-76                [-1, 65, 2]          32,896
          Dropout-77                [-1, 65, 2]               0
      FeedForward-78              [-1, 65, 128]               0
          PreNorm-79              [-1, 65, 128]               0
        LayerNorm-80              [-1, 65, 128]             256
        Rearrange-81                [-1, 2, 65]               0
           Conv2d-82                [-1, 2, 65]         198,144
        Rearrange-83             [-1, 65, 1536]               0
          Softmax-84            [-1, 8, 65, 65]               0
        Rearrange-85                [-1, 2, 65]               0
           Conv2d-86                [-1, 2, 65]          65,664
        Rearrange-87              [-1, 65, 128]               0
          Dropout-88              [-1, 65, 128]               0
        Attention-89              [-1, 65, 128]               0
          PreNorm-90              [-1, 65, 128]               0
        LayerNorm-91              [-1, 65, 128]             256
           Conv2d-92                [-1, 65, 2]          33,024
             GELU-93                [-1, 65, 2]               0
          Dropout-94                [-1, 65, 2]               0
           Conv2d-95                [-1, 65, 2]          32,896
          Dropout-96                [-1, 65, 2]               0
      FeedForward-97              [-1, 65, 128]               0
          PreNorm-98              [-1, 65, 128]               0
        LayerNorm-99              [-1, 65, 128]             256
       Rearrange-100                [-1, 2, 65]               0
          Conv2d-101                [-1, 2, 65]         198,144
       Rearrange-102             [-1, 65, 1536]               0
         Softmax-103            [-1, 8, 65, 65]               0
       Rearrange-104                [-1, 2, 65]               0
          Conv2d-105                [-1, 2, 65]          65,664
       Rearrange-106              [-1, 65, 128]               0
         Dropout-107              [-1, 65, 128]               0
       Attention-108              [-1, 65, 128]               0
         PreNorm-109              [-1, 65, 128]               0
       LayerNorm-110              [-1, 65, 128]             256
          Conv2d-111                [-1, 65, 2]          33,024
            GELU-112                [-1, 65, 2]               0
         Dropout-113                [-1, 65, 2]               0
          Conv2d-114                [-1, 65, 2]          32,896
         Dropout-115                [-1, 65, 2]               0
     FeedForward-116              [-1, 65, 128]               0
         PreNorm-117              [-1, 65, 128]               0
     Transformer-118              [-1, 65, 128]               0
        Identity-119                  [-1, 128]               0
       LayerNorm-120                  [-1, 128]             256
       Rearrange-121                 [-1, 2, 1]               0
          Conv2d-122                 [-1, 2, 1]           1,290
       Rearrange-123                   [-1, 10]               0
================================================================
Total params: 1,989,258
Trainable params: 1,989,258
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.47
Params size (MB): 7.59
Estimated Total Size (MB): 17.07
----------------------------------------------------------------

### Training Logs

![training logs](./images/training_logs.png)



## Notes
- Learning rate sensitivity: Optimal range is 0.01 to 0.005, otherwise accuracy stagnates at 0.1.
- Model head dimension of 4 with batch size of 128 yields better performance and manageable GPU load.
