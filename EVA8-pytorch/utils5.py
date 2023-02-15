
import torch                   #PyTorch base libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F




class Albumentation_cifar_Dataset(Dataset):
  def __init__(self, image_list, train= True):
      self.image_list = image_list
      self.aug = A.Compose({
        A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, value=[0.49139968, 0.48215841, 0.44653091]),
        A.RandomCrop(width = 32,height = 32),
        # A.ShiftScaleRotate(),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, 
                               min_width=8, fill_value=[0.49139968, 0.48215841, 0.44653091], always_apply=False, p = 0.8),
        A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        A.ToGray()

      })

      self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      })
      self.train = train
        
  def __len__(self):
      return (len(self.image_list))

  def __getitem__(self, i):
      
      image, label = self.image_list[i]
      
      if self.train:
        #apply augmentation only for training
        image = self.aug(image=np.array(image))['image']
      else:
        image = self.norm(image=np.array(image))['image']
      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      return torch.tensor(image, dtype=torch.float), label


def info_about_dataset(exp, train = True):
    exp_data = exp.data

    # Calculate the mean and std for normalization
    if train:
        print("[Train")
    else :
        print("Test")
    print(' - Numpy Shape:', exp_data.shape)
    print(' - min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - var:', np.var(exp_data, axis=(0,1,2)) / 255.)



def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))



def visualize_augmented_images(loader, classes, cols = 5, rows = 4):
    images, labels = next(iter(loader))

    sample_size=cols * rows

    images = images[0:sample_size]
    labels = labels[0:sample_size]

    fig = plt.figure(figsize=(10, 10))

    # Show images
    for idx in np.arange(len(labels.numpy())):
        ax = fig.add_subplot(cols, rows, idx+1, xticks=[], yticks=[])
        npimg = images[idx].permute(1,2,0).numpy().astype(dtype=np.float32)
        ax.imshow(npimg, cmap='gray')
        ax.set_title("Label={}".format(classes[labels[idx]]))

    fig.tight_layout()  
    plt.show()

def compute_accuracy_graph(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Epoch vs Accuracy")
    plt.legend()

def compute_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Epoch vs Accuracy")
    plt.legend()

def plot_misclassified_images(model, test_loader, classes, device , cols = 5 ,rows = 4):
    all_misclassified_images = []
    model.eval()
    with torch.no_grad():
        for data , target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    all_misclassified_images.append({'image': data[i], 'predicted_class': classes[pred[i]], 'correct_class': classes[target[i]]})

    fig = plt.figure(figsize=(10,10))
    for i in range(cols * rows):
        sub = fig.add_subplot(cols, rows, i+1)
        misclassified_image = all_misclassified_images[i]
        plt.imshow(misclassified_image['image'].cpu().permute(1,2,0).numpy().squeeze(), cmap='gray', interpolation='none')
        sub.set_title("Correct class: {}\nPredicted class: {}".format(misclassified_image['correct_class'], misclassified_image['predicted_class']))
    plt.tight_layout()
    plt.show()   

'''GradCAM in PyTorch.
Grad-CAM implementation in Pytorch
Reference:
[1] https://github.com/vickyliin/gradcam_plus_plus-pytorch
[2] The paper authors torch implementation: https://github.com/ramprs/grad-cam
'''

layer_finders = {}


def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


@register_layer_finder('resnet')
def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = torch.nn.functional.relu(saliency_map)
        saliency_map = torch.nn.functional.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

def plotGradCAM(net, testloader, classes, device):


    net.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(target[i])
                    predicted_labels.append(pred[i])

    gradcam = GradCAM.from_config(model_type='resnet', arch=net, layer_name='layer4')

    fig = plt.figure(figsize=(10, 10))
    idx_cnt=1
    for idx in np.arange(10):

        img = misclassified_images[idx]
        lbl = predicted_labels[idx]
        lblp = actual_labels[idx]

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = img.unsqueeze(0).to(device)
        org_img = denormalize(img,mean=(0.4914, 0.4822, 0.4471),std=(0.2469, 0.2433, 0.2615))

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(img, class_idx=lbl)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, org_img, alpha=0.4)

        # Show images
        # for idx in np.arange(len(labels.numpy())):
        # Original picture
        
        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(org_img[0].cpu().numpy(),(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title(f"Label={str(classes[lblp])}\npred={classes[lbl]}")
        idx_cnt+=1

        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(heatmap,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("HeatMap".format(str(classes[lbl])))
        idx_cnt+=1

        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(cam_result,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("GradCAM".format(str(classes[lbl])))
        idx_cnt+=1

    fig.tight_layout()  
    plt.show()