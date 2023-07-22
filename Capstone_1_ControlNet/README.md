

## Training and Inferencing Controlnet for Canny Edges 


## References

- https://huggingface.co/blog/controlnet → Detailed blog on Using controlnet pretrained model in stable diffusion pipeline in colab

## Acronyms

* SD - Stable Diffusion
* CTLNET - ControlNet
* CTLNET-CLN - CTLNET cloned. Edited few modules to enable it to run in colab. [ctlnet-custom]
* CDS - Custom DataSet

## Overview

- ControlNet is a neural network structure to control diffusion models by adding extra conditions.
- For example, let us say we want to create winged drones flying like birds from a given image.
- SD with CTLNET can help us achieve that as shown below:

 ![a3-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/d44ea153-e4dd-404c-82c3-020b2457ade3)
 
- In short, CTLNET equips us with the ability to control SD and generate images in the way we desire by passing a control image and prompt.
- In the above example, we are converting the input image having 2 birds by passing the control image (canny edge black & white image) and prompt "A Winged Drone" to get back 2 winged drones flying similarly like the birds in input image.

- Overall architecture of CTLNET is as follows:

![CTLNET_Architecture](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/02dbaad1-70d7-4cef-9f0f-bf84f1ce22e0)

- The "trainable" one (with blue shades) learns our control condition (eg: canny edge, pose etc.). The "locked" one preserves the model.
- Thanks to CTLNET, training with small custom datasets will not destroy the production-ready SD models.

## Approach
- Training is done based on [ctlnet-train] 
- In this repo, we are attempting to **train CTLNET Canny Edge condition from scratch using a CDS**.
- Objective : Given an input image, its canny edge sketch and a prompt, get back an output image 
    -  Having the object mentioned in the prompt 
    -  With edges resembling the canny edge sketch 
    -  And overall canvas resembling the input image.
- CDS is created in the same format as [Fill50K] dataset.
- CDS is created from 7574 flying bird RGB images that were downloaded from internet (creative common license images).
- CDS is driven via prompt.json - a sample json record shown as below 
    - {"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"}
    - Here **source** → Control image (in our case Canny Edge Sketch).This was created using cv2.canny.
        
        ![source](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/90ec02f4-b32e-40a9-8fcf-6b34455f2654)
        
    - **target** → Original image
        
        ![target](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4a99e4a6-fe45-427e-97a6-55e441fdd435)
        
    - **prompt** → A caption explaining the image.This was created using BLIP captioning.

        {"source": "source.jpg", "target": "target.jpg", "prompt": "two birds flying in the sky"}

## Attempts-and-Results

- Attempt

    - Reference notebook : [S15_ControlNet_V6.ipynb ](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V6.ipynb) 
    - Preloaded the model with v1-5-pruned.ckpt (no canny edge involved so far).
        ```
        downloaded_model_path = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5",
                                        filename="v1-5-pruned.ckpt",
                                        use_auth_token=True)

        model = create_model(config_path='./models/cldm_v15.yaml')

        pretrained_weights = torch.load(input_path)
        ```
    - Initialized the control-model weights (trainable copy) of the model.
        ```
        # Get the state dictionary of the model
        state_dict = model.state_dict()

        # Manually initialize each parameter tensor of control_model (trainable conditioning block)
        i = 0
        for name, param in state_dict.items():
            if 'control_model' in name:
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
            i += 1

        # Load the updated state dictionary into the model
        model.load_state_dict(state_dict)
        print(f'Total parms : {i}')
        ```

    - Trained 7574 flyingbirds images present in CDS & ran 5 epochs for batch_size = 8 (1250 batches) against A100 High-RAM GPU.
    - Results were good.
        - Sample created

            ![a7-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/539e9199-e001-4567-b8a9-1e5bf5babf00)
        - Reconstructed image

            ![a7-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/fa46a625-1529-4c58-9f26-a28dcb3e1245)
        - Control image given as input for conditioning

            ![a7-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/a41f4cfb-1233-435c-a7c1-2eff49d35ea4)
        - Captions supplied

            ![a7-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/7f0c42e7-e53a-415c-9ff3-6298e19bb697)  
    - Inferred with the same model & results are promising

        ![a7-inference-1](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/c40a90f6-c296-40bf-940a-eafe3a970c18)

        ![a7-inference-2](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4907326e-b5d7-449b-9875-4910a03f9e7f)

        ![a7-inference-3](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/b8937fbb-2b68-4cb8-8bd3-15453da19627)
