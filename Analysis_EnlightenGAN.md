# EnlightenGAN: Deep Light Enhancement without Paired Supervision
## Intro
* Proposed a highly effective unsupervised generative adversarial network, dubbed **EnlightenGAN**, that can be trained without low/normal-light image pairs, yet proves to generalize very well on various real-world test images.
* We propose to regularize the unpaired training using the information extracted from the input itself, and benchmark a series of innovations for the low-light image enhancement problem, including a global-local discriminator structure, a self-regularized perceptual loss fusion, and attention mechanism. 
* code link: https://github.com/VITA-Group/EnlightenGAN (official)   others: https://paperswithcode.com/paper/enlightengan-deep-light-enhancement-without#code
* IEEE Transaction on Image Processing, 2020, [EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)
* * * 
## Overall Architecture
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/EnlightenGAN.png" width="110%">
#### Innovations
* 1. EnlightenGAN is the first work that successfully introduces unpaired training to low-light image enhancement. Such a training strategy removes the dependency on paired training data and enables us to train with larger varieties of images from different domains.
It also avoids overfitting any specific data generation protocol or imaging device that previous works implicitly rely on, hence leading to notably improved real-world generalization.
* 2. EnlightenGAN gains remarkable performance by imposing  a global-local discriminator structure that handles spatially-varying light conditions in the input image; (ii) the idea of self-regularization, implemented by both the self feature preserving loss and the self-regularized attention mechanism. The selfregularization is critical to our model success, because of the unpaired setting where no strong form of external supervision is available.
* 3. EnlightenGAN is compared with several state-of-theart methods via comprehensive experiments. The results are measured in terms of visual quality, noreferenced image quality assessment, and human subjective survey. All results consistently endorse the superiority of EnlightenGAN. Morever, in contrast to existing paired-trained enhancement approaches, EnlightenGAN proves particularly easy and flexible to be adapted to enhancing real-world low-light images from different domains.








### Representitive Results



## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`

Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict```

### Dataset preparing

Training data [[Google Drive]](https://drive.google.com/drive/folders/1fwqz8-RnTfxgIIkebFG2Ej3jQFsYECh0?usp=sharing) (unpaired images collected from multiple datasets)

Testing data [[Google Drive]](https://drive.google.com/open?id=1PrvL8jShZ7zj2IC3fVdDxBY1oJR72iDf) (including LIME, MEF, NPE, VV, DICP)

And [[BaiduYun]](https://github.com/TAMU-VITA/EnlightenGAN/issues/28) is available now thanks to @YHLelaine!


If you find this work useful for you, please cite
```
@article{jiang2019enlightengan,
  title={EnlightenGAN: Deep Light Enhancement without Paired Supervision},
  author={Jiang, Yifan and Gong, Xinyu and Liu, Ding and Cheng, Yu and Fang, Chen and Shen, Xiaohui and Yang, Jianchao and Zhou, Pan and Wang, Zhangyang},
  journal={arXiv preprint arXiv:1906.06972},
  year={2019}
}
```

## Dataset Preview
*

| image | Low light image | Normal light image |
| :----: | :----: | :----: |
| detail | 600 * 400 pixel (width * height), created by Adobe Light-room, .png format | 600 * 400 pixel (width * height), created by Adobe Light-room, .png format |
|image pair e.g.| <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/lowlightimg2.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/2.png"> |
* * * 
### I. Decom-Net 
* It takes in pairs of low/normal-light images at the training stage, while only low-light images as input at the training at the testing stage.
* With the constraints that the low/normal-light images share the same reflectance and the smoothness of illumination, Decom-Net learns to extract the consistent *R* between variously illuminated images in a data-driven way.
* During training, there is no need to provide the ground truth (means Correct Label) of the reflectance and illumination. Only requisite knowledge including the consistency of reflectance and the smoothness of illumination map is embedded into the network as loss functions. 
* Thus, the decomposition of our network is automatically learned from paired low/normal-light images, and by nature suitable for depicting the light variation among the images under different light conditions.
#### Decom-Net Interpretation
1. Decom-Net takes the low-light image *Slow* and the normal-light one *Snormal* as input, then estimates the reflectance *Rlow* and the illumination *Ilow* for *Slow*, as well 
as *Rnormal* and *Inormal* for *Snormal*, respectively. 
2. It uses several 3×3 convolutional layers with Rectified Linear Unit (ReLU) as the activation function are followed to map the RGB image into reflectance and illumination. A 
3×3 convolutional layer projects *R* and *I* from feature space, and sigmoid function is used to constrain both *R* and *I* in the range of [0, 1].
* * *
### Loss Functions in DecomNet
* **1. The loss *L* consists of 3 terms: reconstruction loss *Lrecon*, invariable reflectance loss *Lir*, and illumination smoothness loss *Lis*:**
    > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/loss.png" width='80%'>
    > * where *λir* and *λis* denote the coefficients to balance the consistency of reflectance and the smoothness of illumination.

*  **1.1 The *Lrecon* is defined as:**
    > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/lossRecon.png" width='90%'>
    > * The formula means that *Lrecon* equals to the sum of (coefficients muliply the L1 Norm of *Ri* element-wise multiply *Ij* minus *Sj*), where *i* and *j* are low and normal index, where *Ri* * *Ij* is the reconstruct image and *Sj* is the original image. When we minimize the loss function, we minimize the difference between predicted image 
and real image, thus obtain the best reconstruction performance.
    > * Subscripts in *i = low, normal* means this *Lrecon* calculation formula works on both low and normal light images.
    > * In later code, we can see it calculates loss between low and low, normal and normal, and mutual losses between low and normal.
* **Q: Why we use L1 Norm here?**
    > * **A: To sparse the weights, thus we can complete feature selection and add model interpretability. And if compared with L2 norm, L1 create less features and minimize the            weights much faster than L2; L1 is also Robust to abnormal values.**
* **1.2 Invariable reflectance loss *Lir* is introduced to constrain the consistency of reflectance:**
    > * This is a loss function in DecomNet.
    > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/LossInvariableReflectance.png" width="70%">
    > * **My comments: The author mentioned that low light and normal images should share the same reflectance in any conditions because reflectance is the intrinsic property of objects. However, there might be color differences between normal and low light conditions. And that's why we need to minimize this constraint *Lir* to ensure image pairs have same Reflectance before the next step--Enhance-Net. This is the what Retinex theory indicates.**

* **1.3 The *Lis* is defined as:**
    > * This is a loss function in DecomNet.
    > * Total variation minimization (TV), which minimizes the gradient of the whole image, is often used as smoothness prior for various image restoration tasks. However, directly using TV as loss function fails at regions where the image has strong structures or where lightness changes drastically. It is due to the uniform reduction for gradient of illumination map regardless of whether the region is of textual details or strong boundaries. In other words, TV loss is structure-blindness. The illumination is blurred and strong black edges are left on reflectance, as illustrated below.
    > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/TV.png" width="90%">
    > * To make the loss aware of the image structure, the original TV function is weighted with the gradient of reflectance map. The final Lis is formulated as:
    > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/LossIlluminationSmoothness.png" width="80%">
    > * where *∇* denotes the gradient including *∇h (horizontal)* and *∇v (vertical)*, and *lg* denotes the coefficient balancing the strength of structure-awareness. With the weight *exp(−lg∇Ri)*, *Lis* loosens the constraint for smoothness where the gradient of reflectance is steep, in other words, where image structures locate and where the illumination should be discontinuous. Subscripts in *i = low, normal* means this *lis* calculation formula works on both low and normal light images. And by calculating gradients of R in order to allocate weights to I gradients, it enables area in I as smooth as possible where corresponds to smooth area in R.
    > * Our structure-aware smoothness loss is weighted by reflectance.
    > * Calculations of gradients are implemented both horizontally and vertically, I think this is because we need to consider all directions in the Reflectance, in other words, it is similar to a traversal of an image gradients to detect where structures are obvious.

* * * 
# Q & A
>
#### Data preprocessing:
1. Collect synthetic image pairs from raw images, they transform images into YCbCr channel and calculate the histogram of Y channel.
2. The image pairs contains low light and normal light images in the same angle, and all resolutions are the same, 600*400 pixels (width*height), depth is 24, .png format, created by Adobe Light-room.
#### Intro to Decom-Net: 
1. It takes 5 convolutional layers with a ReLU activation function between to ConV2d layers without ReLU. 
2. In the results, low/normal-light images share the same reflectance. And the illumination map should be smooth but retain main structures, which is obtained by a structure-aware total variation loss.
#### Intro to Enhance-Net:
It adjusts the illumination map to maintain consistency at large regions while tailor local distributions by multi-scale concatenation.
It consists of 3 down-sampling blocks and 3 up-sampling ones.
1. Train both Net,
2. Finetune the net end-to-end using SGD (Stochastic Gradient Descent) with back-propagation.
3. Batch size = 16, patch_size = 96*96, learning rate = 0.001, 0.1 and 10


### Q1. Why the channel number of the input Conv layer in DecomNet is 4?
* A: 3 RGB channels represent Reflectance and 1 illumination channel, this illumination channel is calculated from max values in RGB channel

### Q2. Why using 64 channel in layers?
* A:  64 is an empirical value. Trial-and-error process results in 64.







