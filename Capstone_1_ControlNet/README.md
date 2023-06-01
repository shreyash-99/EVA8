# Training Control-Net For Controlling Stable Diffusion

#### Training Data
Used 300 random classes from ImageNet dataset, in total 10,521 pictures were used.

### for training 
Control net provides direct access to models in whihc stable diffusion part of the model is fixed and the control net part of model can be trained independantly. It requires data in a form of dictionery with 3 arguments: 
1. The original image
2. The canny image which is derived using the the opencv library
3. A prompt is needed which is derived for every photo using BLIP model

This dictionary can then be passed to the control net training function for training the model, havent trained till th eend as it was not possible on colab free account. I am a college student and i think the result will be good only so i didnt wanted to buy colab units.  
