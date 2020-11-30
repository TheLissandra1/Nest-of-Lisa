# Learning PyTorch Basic
## Convolutional Neural Network
#### Intro
* CNN is similar to ordinary neural networks and they are made up of neurons that have learned weights and biases.
* The main difference is that CNN mainly work on image data and apply these so-called convolutional filters so a typical CNN contains: ConV + activation function (e.g. ReLU) and followed by a Pooling layer.
* These layers are used to automatically learn some features from the images and then, at the end, we have one or more fully connected layers for actual classification tasks.
#### Layers
1. Convolutional layer
2. Max pooling layer: used to down-sampling, reduce computational costs by reducing the size of the image so this reduces the number of parameters our model has to learn. And it helps to avoid overfitting by providing an abstracted form of the input.
#### Code Example
*
```python 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
           
           
```
