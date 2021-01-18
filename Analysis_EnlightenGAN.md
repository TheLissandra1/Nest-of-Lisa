# EnlightenGAN: Deep Light Enhancement without Paired Supervision
## Intro
* Proposed a highly effective unsupervised generative adversarial network, dubbed **EnlightenGAN**, that can be trained without low/normal-light image pairs, yet proves to generalize very well on various real-world test images.
* We propose to regularize the unpaired training using the information extracted from the input itself, and benchmark a series of innovations for the low-light image enhancement problem, including a global-local discriminator structure, a self-regularized perceptual loss fusion, and attention mechanism. 
* code link: https://github.com/VITA-Group/EnlightenGAN (official)   others: https://paperswithcode.com/paper/enlightengan-deep-light-enhancement-without#code
* IEEE Transaction on Image Processing, 2020, [EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)
* * * 
### Representitive Results
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Repre_Results.png" width="110%">
## Overall Architecture
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/EnlightenGAN.png" width="110%">

#### Notable Innovations
* 1. EnlightenGAN is the first work that successfully introduces unpaired training to low-light image enhancement. Such a training strategy removes the dependency on paired training data and enables us to train with larger varieties of images from different domains.
It also avoids overfitting any specific data generation protocol or imaging device that previous works implicitly rely on, hence leading to notably improved real-world generalization.
* 2. EnlightenGAN gains remarkable performance by imposing  a global-local discriminator structure that handles spatially-varying light conditions in the input image; the idea of self-regularization, implemented by both the self feature preserving loss and the self-regularized attention mechanism. The self-regularization is critical to the model success, because of the unpaired setting where no strong form of external supervision is available.
* 3. EnlightenGAN is compared with several state-of-the-art methods via comprehensive experiments. The results are measured in terms of visual quality, noreferenced image quality assessment, and human subjective survey. All results consistently endorse the superiority of EnlightenGAN. Morever, in contrast to existing paired-trained enhancement approaches, EnlightenGAN proves particularly easy and flexible to be adapted to enhancing real-world low-light images from different domains.


## Related Works
#### Paired Datasets
* LOL Dataset
#### Traditional Approaches
* LIME: 
#### Deep Learning Approaches
* LL-Net:
* HDR-Net:
#### Adversarial Learning 


## Method
### 3.1 Global-Local Discriminators
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/12.png" width="50%">
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/34.png" width="50%">
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/56.png" width="50%">
### 3.2 Self Feature Preserving Loss
 > * In unpaired setting, we propose to instead constrain the VGG feature distance between the input low-light and its enhanced normal-light output. 
 > * Based on the author's empirical observation, the classification results by VGG models are not very sensitive when manipulating the input pixel intensity range.
 > * We call it *self feature preserving loss* to stress its self-regularization utility to preserve the image content features to itself, before and after the enhancement. 
* The self feature preserving loss *Lsfp* is defined as:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/7Lsfp.png" width="50%">
* where 
    - IL denotes the input low-light image.
    - G(IL) denotes the generator's enhanced output. 
    - *Φi,j* denotes the feature map extracted from a VGG-16 model pre-trained on ImageNet.
    - i represents its i-th maxpooling
    - j represents its j-th convonlutional layer after i-th maxpooling layer.
    - Wi,j and Hi,j are the dimensions of the extracted feature maps.
    - By default we choose i=5, j=1.
* For the local discriminator, the cropped local patches from input and output images are also regularized by a similarly defined self feature preserving loss, *Lsfp_local* .
* We add an instance normalization layer after the VGG feature maps before feeding into *Lsfp* and *Lsfp_local* to stablize training.
* The overall loss function for training EnlightenGAN is thus written as:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/8Loss.png" width="50%">
### 3.3 U-Net Generator Guided with Self-Regularized Attention
* U-Net has achieved huge success on semantic segmentation, image restoration and enhancement. By extracting multi-level features from different depth layers. U-Net preserves rich texture information and synthesizes high quality images using multi-scale context information. We adopt U-Net as our generator backbone.
* Unet Architecture:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Unet.png" width="70%">
* Intuitively, in a low-light image of spatially varying light condition, we always want to enhance the dark regions more than bright regions, so that the output image has neither over- nor under-exposure.
* We take the illumination channel *I* of the input RGB image, normalize to [0,1], and then use *1-I* (element-wise difference) as our self-regularized attention map.
* We then resize the attention map to fit each feature map and multiply it with all intermediate feature maps as well as the output image.
* We emphazie that our attention map is also a form of self-regularization, rather than learned with supervision. Despite its simplicity, the attention guidance shows to improve the visual quality consistency.

* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/EnlightenGAN.png" width="110%">
* Our attention-guided U-Net generator is implemented with 8 ConV blocks. Each block consists of two 3 x 3 ConV layers, followed by LeakyReLu and a batch normalization layer. At the upsampling stage, we replace the standard deconvolutional layer in U-Net with one bilinear upsampling layer + one ConV layer, to mitigate the checkerboard artifacts. The final architecture of EnlightenGAN is illustrated in the image above.
## Experiments
### 4.1 Dataset and Implementation Details
* We assemble a mixture of 914 low-light images and 1016 normal light images from several dataset and also HDR sources without the need to keep any pair.
#### Training Dataset Preview
*
| images | Low light image | Normal light image |
| :----: | :----: | :----: |
| detail | 600 * 400 pixel (width * height) .png format | 600 * 400 pixel (width * height), .png format |
|image e.g.| <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/105_2.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/normal00101.png"> |

#### Testing Dataset Preview
* We choose standard ones used in previous works (NPE, LIME, MEF, DICM, VV, etc.).
* Noted: NPE dataset image formats and size are various.
*
| images | NPE | LIME | MEF | DICM | VV |
| :----: | :----: | :----: | :----: | :----: | :----: |
| detail | 750 * 725 pixel (width * height) .jpg format | 720 * 680 pixel (width * height) .bmp format | 512 * 340 pixel (width * height) .png format | 480 * 640 pixel (width * height) .JPG format | 2304* 1728 pixel (width * height) .jpg |
|image e.g.| <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/birds.jpg"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/1.bmp"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/A.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/01.JPG"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/P1000205.jpg"> |
* EnlightenGAN is first trained from the scratch for 100 epochs with the learning rate of 1e-4, followed by another 100 epochs with the learning rate linearly decayed to 0. We use the Adam optimizer and the batch size is set to be 32.
* Thanks to the lightweight design of one-path GAN without using cycle-consistency, the training time is much shorter than cycle based methods. The whole training process takes 3 hours on 3 Nvidia 1080Ti GPUs.


### 4.2 Ablation Study
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig3.png" width="110%">
### 4.3 Comparison with State-of-the-Arts
#### 4.3.1 Visual Quality Comparison
#### 4.3.2 Human Subjective Evaluation 
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig4.png" width="110%">
#### 4.3.3 No-Referenced Image Quality Assessment
### 4.4 Adaptation on Real-World Images
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig6.png" width="110%">
### 4.5 Pre-Processing for Improving Classification

## Conclusion
* In this paper, we address the low-light enhancement problem with a novel and flexible unsupervised framework.
* The proposed EnlightenGAN operates and generalizes well without any paired training data. The experimental results on various low light datasets show that our approach outperforms multiple state-of-the-art approaches under both subjective and objective metrics.
* Furthermore, we demonstrate that EnlightenGAN can be easily adapted on real noisy lowlight images and yields visually pleasing enhanced images.
* Our future work will explore how to control and adjust the light enhancement levels based on user inputs in one unified model. Due to the complicacy of light enhancement, we also expect integrate algorithm with sensor innovations.












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



* * * 
# Q & A
### Q1. Why the channel number of the input Conv layer in DecomNet is 4?
* A: 3 RGB channels represent Reflectance and 1 illumination channel, this illumination channel is calculated from max values in RGB channel

### Q2. Why using 64 channel in layers?
* A:  64 is an empirical value. Trial-and-error process results in 64.







