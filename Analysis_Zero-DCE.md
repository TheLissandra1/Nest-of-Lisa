# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
* * *
## 1. Intro
#### 1. Zero-DCE is non-reference, i.e. is independent of paired and unpaired training data, thus avoid the risk of overfitting.
   1. Non-reference is achieved by a set of specially designed non-reference loss functions.
   2. The loss function consider multiple factors of light enhancement, including spatial consistency loss, exposure control loss, color constancy loss, and illumination smoothness loss.
#### 2. The author designs image-specific curve that is able to approximate pixel-wise and higher-order curves by iteratively applying itself.
   1. High-order curves are pixel-wise adjustment on the dynamic range of input to obtain an enhanced image as these curves maintain the range of enhanced image and remain contrast of neighboring pixels.
   2. Curves are differentiable so that we can learn adjustable parameters of curves through a CNN.
#### 3. Zero-DCE supersedes state-of-art performance both qualitatively and quantitatively, e.g. CNN-based method and EnlightenGAN.
#### 4. Zero-DCE is capable of improving high-level visual tasks, e.g. face detection, and it is lightweight.
* * * 
## 2. Related Work

## 3. Methodology
### 3.0 Framework
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_LIME_Zero_DCE/DCENetFramework.png", width="80%">
### 3.1 Light-Enhancement Curve (LE-Curve) 💜
   > * This self-adaptive curve parameters only depend on the input image.
   > * Three objectives: 1) each pixel value of the enhanced image should be in the normalized range of [0,1] to avoid information loss induced by overflow truncation; 2) this curve should be monotonous to preserve the differences (contrast) of neighboring pixels; 3) the form of this curve should be as simple as possible and  differentiable in the process of gradient backpropagation.
   > * Curve: 
   >   <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_LIME_Zero_DCE/LECurve.png', width="80%">
   > *
### 3.2 DCE-Net

**Code Interpretation
```python
import torch
import torch.nn as nn
import math
import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):  #创建增强网络结构

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True) 

		number_f = 32  #通道为32个
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)       #创建7个卷积层

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)   #最大池化层
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)   #用于2D数据的线性插值算法  
		
	def forward(self, x):   

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))  #在维度1上拼接x3,x4
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)  ##在维度1上进行划分，每大块包含3个小块


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r
```

### 3.3 Non-Reference Loss Functions 💜 
##### Spatial Consistency Loss
##### Exposure Control Loss
##### Color Constancy Loss
##### Illumination Smoothness Loss
#### Total Loss

## 4. Experiments
### 4.1 Ablation Study
##### Contribution of Each Loss
##### Effect of Parameter Settings
##### Impact of Training Data

### 4.2 Benchmark Evaluations
##### Visual and Perceptual Comparisons
##### Quantitative Comparisons
##### Face Detection in the Dark

## 5. Conclusion


## Q & A 
**Q1: How does monotonous LE-Curve preserve differences of neighboring pixels?**
    * A:

