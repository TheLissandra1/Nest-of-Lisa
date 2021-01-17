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
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/7Lsfp.png" width="50%">
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/8Loss.png" width="50%">
### 3.3 U-Net Generator Guided with Self-Regularized Attention


## Experiments
### 4.1 Dataset and Implementation Details
### 4.2 Ablation Study
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig3.png" width="110%">
### 4.3 Comparison with State-of-the-Arts
#### 4.3.1 Visual Quality Comparison
#### 4.3.2 Human Subjective Evaluation 
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig5.png" width="110%">
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

## Dataset Preview
*

| image | Low light image | Normal light image |
| :----: | :----: | :----: |
| detail | 600 * 400 pixel (width * height), created by Adobe Light-room, .png format | 600 * 400 pixel (width * height), created by Adobe Light-room, .png format |
|image pair e.g.| <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/lowlightimg2.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/2.png"> |



* * * 
# Q & A
### Q1. Why the channel number of the input Conv layer in DecomNet is 4?
* A: 3 RGB channels represent Reflectance and 1 illumination channel, this illumination channel is calculated from max values in RGB channel

### Q2. Why using 64 channel in layers?
* A:  64 is an empirical value. Trial-and-error process results in 64.







