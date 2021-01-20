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
* LIME: illumination of each pixel was first estimated by finding the maximum value in its RGB channels, then the illumination map was constructed by imposing a structure prior.
#### Deep Learning Approaches
* LL-Net: a stacked auto-encoder to learn joint denoising and low-light enhancement on the patch level.
* Retinex-Net: provided an end-to-end framework to combine the Retinex theory and deep networks.
* HDR-Net: incorporated deep networks with the ideas of bilateral grid processing and local affine color transforms with pairwise supervision.
* Learning to see in the dark: 
#### Adversarial Learning 


## Method
### 3.1 Global-Local Discriminators
* To enhance local regions adaptively in addition to improving the light globally, we propose a novel global-local discriminator structure, both using PatchGAN for real/fake discrimination.
* In addition to the image-level global discriminator, we add a local discriminator by taking randomly cropped local patches from both output and real normallight images, and learning to distinguish whether they are real (from real images) or fake (from enhanced outputs). Such a global-local structure ensures all local patches of an enhanced images look like realistic normal-light ones, which proves to be critical in avoiding local over- or underexposures as our experiments will reveal later.
* Furthermore, for the global discriminator, we utilize the recently proposed relativistic discriminator structure which estimates the probability that real data is more realistic than fake data and also directs the generator to synthesize a fake image that is more realistic than real images.
* The standard function of relativistic discriminator is:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/12.png" width="50%">
* where C denotes the network of discriminator, xr and xf are sampled from the real and fake distribution, σ represents the sigmoid function. We slight modify the relativistic discriminator to replace the sigmoid function with the leastsquare GAN (LSGAN) [36] loss. 
* Finally, the loss functions for the global discriminator D and the generator G are:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/34.png" width="50%">
* For the local discriminator, we randomly crop 5 patches from the output and real images each time. Here we adopt the original LSGAN as the adversarial loss, as follows:
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
* To demonstrate the effectiveness of each component proposed in Sec. 3, we conduct several ablation experiments. Specifically, we design two experiments by removing the components of local discriminator and attention mechanism, respectively. 
* As shown in Fig. 3, the first row shows the input images. The second row shows the image produced by EnlightenGAN with only global discriminator to distinguish between low-light and normal-light images. The third row is the result produced by EnlightenGAN which does not adopt self-regularized attention mechanism and
uses U-Net as the generator instead. The last row is produced by our proposed version of EnlightenGAN. The enhanced results in the second row and the third row tend to contain local regions of severe color distortion or under-exposure, namely, the sky over the building in Fig.3(a), the roof region in Fig.3(b), the left blossom in Fig.3(c), the boundary of tree and bush in Fig.3(d), and the T-shirt in Fig.3(e). 
* In contrast, the results of the full EnlightenGAN contain realistic color and thus more visually pleasing, which validates the effectiveness of the global-local discriminator design and self-regularized attention mechanism. More images are in the supplementary materials.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig3.png" width="110%">
### 4.3 Comparison with State-of-the-Arts
* In this section we compare the performance of EnlightenGAN with current state-of-the-art methods. We conduct a list of experiments including visual quality comparison, human subjective review and no-referenced image quality assessment (IQA), which are elaborated on next.
#### 4.3.1 Visual Quality Comparison
* We first compare the visual quality of EnlightenGAN with several recent competing methods. The results are demonstrated in Fig. 4, where the first column shows the original low-light images, and the second to fifth columns are the images enhanced by: a vanilla CycleGAN [9] trained using our unpaired training set, RetinexNet [5], SRIE [20], LIME [21], and NPE [19]. The last column shows the results produced by EnlightenGAN.
* We next zoom in on some details in the bounding boxes. LIME easily leads to over-exposure artifacts, which makes the results distorted and glaring with the some information missing. The results of SRIE and NPE are generally darker compared with others. CycleGAN and RetinexNet generate unsatisfactory visual results in terms of both brightness and naturalness. In contrast, EnlightenGAN successfully not only learns to enhance the dark area but also preserves the texture details and avoids over-exposure artifacts.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig4.png" width="110%">
* Figure 4: Comparison with other state-of-the-art methods. Zoom-in regions are used to illustrate the visual difference. 
    - First example: EnlightenGAN successfully suppresses the noise in black sky and produces the best visible details of yellow wall.
    - Second example: NPE and SRIE fail to enhance the background details. LIME introduces over-exposure on the woman’s face. However, EnlightenGAN not only restores the background details but also avoids over-exposure artifacts, distinctly outperforming other methods. 
    - Third example: EnlightenGAN produces a visually pleasing result while avoiding overexposure artifacts in the car and cloud. Others either do not enhance dark details enough or generate over-exposure artifacts. Please zoom in to see the details.
#### 4.3.2 Human Subjective Evaluation 
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/comparison.png" width="110%">

#### 4.3.3 No-Referenced Image Quality Assessment
* We adopt Natural Image Quality Evaluator (NIQE) [48], a well-known no-reference image quality assessment for evaluating real image restoration without ground-truth, to provide quantitative comparisons. The NIQE results on five publicly available image sets used by previous works (MEF, NPE, LIME, VV, and DICM) are reported in Table 1: a lower NIQE value indicates better visual quality. EnlightenGAN wins on three out of five sets, and is the best in terms of overall averaged NIQE. This further endorses the superiority of EnlightenGAN over current state-of-the-art methods in generating high-quality visual results.
* <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/table1.png" width = "60%">
* Figure 5: The result of five methods in the human subjective evaluation. In each histogram, x-axis denotes the ranking index (1 ∼ 5, 1 represents the highest), and y-axis denotes the number of images in each ranking index. EnlightenGAN produces the most top-ranking images and gains the best performance with the smallest average ranking value.
### 4.4 Adaptation on Real-World Images
* Domain adaptation is an indispensable factor for realworld generalizable image enhancement. The unpaired training strategy of EnlightenGAN allows us to directly
learn to enhance real-world low-light images from various domains, where there is no paired normal-light training data or even no normal-light data from the same domainavailable. We conduct experiments using low-light images from a real-world driving dataset, Berkeley Deep Driving (BBD-100k) [1], to showcase this unique advantage of EnlightenGAN in practice.
* We pick 950 night-time photos (selected by mean pixel intensity values smaller than 45) from the BBD-100k set as the low-light training images, plus 50 low-light images for hold-out testing. Those low-light images suffer from severe artifacts and high ISO noise. We then compare two EnlightenGAN versions trained on different normal-light image sets, including: 
    - 1) the pre-trained EnlightenGAN model as described in Sec. 4.1, without any adaptation for BBD-100k; 
    - 2) EnlightenGAN-N: a domain-adapted version of EnlightenGAN, which uses BBD-100k low-light images from the BBD-100k dataset for training, while the normal-light images are still the high-quality ones from our unpaired dataset in Sec. 4.1. We also include a traditional method, Adaptive histogram equalization (AHE), and a pretrained LIME model for comparison.
* As shown in Fig. 6, the results from LIME suffer from severe noise amplification and over-exposure artifacts, while AHE does not enhance the brightness enough. The
original EnlightenGAN also leads to noticeable artifacts on this unseen image domain. In comparison, EnlightenGANN produces the most visually pleasing results, striking an impressive balance between brightness and artifact/noise suppression. Thanks to the unpaired training, EnlightenGAN could be easily adapted into EnlightenGAN-N without requiring any supervised/paired data in the new domain, which greatly facilitates its real-world generalization.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig6.png" width="110%">
* Figure 6: Visual comparison of the results on the BBD-100k dataset [1]. EnlightenGAN-N is the domain-adapted version of EnlightenGAN, which generates the most visually pleasing results with noise suppressed. Please zoom in to see the details.

### 4.5 Pre-Processing for Improving Classification
* Image enhancement as pre-processing for improving subsequent high-level vision tasks has recently received increasing attention [28, 49, 50], with a number of benchmarking efforts [47, 51, 52]. We investigate the impact of light enhancement on the extremely dark (ExDark) dataset [53], which was specifically built for the task of low-light image recognition. The classification results after light enhancement could be treated as an indirect measure on semantic information preservation, as [28, 47] suggested.
* The ExDark dataset consists of 7,363 low-light images, including 3000 images in training set, 1800 images in validation set and 2563 images in testing set, annotated into 12 object classes. We use its testing set only, applying our pretrained EnlightenGAN as a pre-processing step, followed by passing through another ImageNet-pretrained ResNet-50 classifier. Neither domain adaption nor joint training is per-formed. The high-level task performance serves as a fixed semantic-aware metric for enhancement results.
* In the low-light testing set, using EnlightenGAN as pre-processing improves the classification accuracy from 22.02% (top-1) and 39.46% (top-5), to 23.94% (top-1) and
40.92% (top-5) after enhancement. That supplies a side evidence that EnlightenGAN preserves semantic details, in addition to producing visually pleasing results. We also conduct experiment using LIME and AHE. LIME improves the accuracy to 23.32% (top-1) and 40.60% (top-5), while AHE obtains to 23.04% (top-1) and 40.37% (top-5).
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
### Q1. Why 
* A: 






