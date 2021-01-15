# Low-light Image Enhancement Algorithm Based on Retinex and Generative Adversarial Network
* * *
## 1. Intro
* * * 
## 2. Background & Previous Work
* A. Low-light Enhancement and Image Denoising
  > 1. Histogram Equalization (HE)
  > 2. Contrast-Limiting Adaptive Histogram Equalization (CLAHE)
  > 3. Unsharp Masking (UM)
  > 4. Multiscale Retinex (MSR)
  > 5. LIME
  > 6. LLNET etc.
* B. GAN: The generative network is trained for generating realistic synthetic samples from a noise distribution to cheat the discriminative network. The discriminative network aims to distinguish true and generated fake samples. In addition to random sample from a noise distribution, various form data can also be used as input to the generator.

## 3. Proposed Research
### A. Network Structure  💜
   > * Based on Retinex theory.
   > * <img src = "https://github.com/TheLissandra1/Nest-of-Lisa/blob/master/ImageLinks_Retinex-GAN/Retinex_Theory.png" width = "20%">
   > * Pipeline of Retinex-GAN:
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Retinex-GAN.png" width="110%">
   > * Retinex-GAN: x is low-light image and y is the corresponding ground truth image. The generative network is composed of two parallel Unets to split
x and y into reflected image when data training starts, which is also termed as the decomposition process. The following enhancement process in the
generative network is responsible for generating the reflected images for x by the yellow Unet and then a new image is formed by combining the reflected
part and illumination part. At last, the discriminative network, actually a normal convolutional neural network, is to distinguish x and y. When testing, only
the area encircled by red line rectangular works.
   > * Unet: 
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Unet.png" width="50%">
   
### B. Regularization Loss 💜
   > * We assume that RGB channels of images are exposed to the same level of light (Retinex), thus we use the Unet to split the image S into reﬂected image R with three channels and illumination image I with one channel. It has proved that the assumption fails to maximize the network performance while illumination image I and reﬂected image R can’t reconstruct the original image S effectively (in a paper named 'Lightness and retinex theory' written by E. H. Land and J. J. McCann. in 1971). Then we try the second strategy. The author assume that the illumination image is also **three channels**. On the basis of this assumption, there is a serious problem that the network quickly falls into a local optimal solution. 
   > * Brief analysis of local optimal solution:
   > * Given 2 matrix S1 and S2, if we want to optimize min|R1-R2| to satisfy the following equation for all points i, j:
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimizeR1-R2.png" width="30%">
   > * The best two solutions are as below；
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimal solution1.png" width="30%">
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimal solution2.png" width="30%">
   > * We can see that after many iterations, the value of reﬂected image will become all 1 or -1 and the illumination image will be same as original or the reversed image. This means that the decomposition is meaningless. To solve this problem, the author propose a regularization loss *Lreg* which can prevent the RGB values of generated illumination image from approaching 1 or -1 to avoid that the network falls into a local optimal solution.
   > * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lregularization.png" width="40%">

* * *
### C. Multitask Loss 💜 
   > * Total loss (Multi-task loss) is defined as: 
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Multitask Loss.png" width = "50%">
   > * where λrec, λdec, λcom, and λcCAN are the loss of Pix2pix-GAN which includes the LcGAN loss and the L1 loss while the L1 loss is replaced by smoothL1 loss.
   > * Pix2pix-GAN:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Pix2pix.png" width = "30%">
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Pix2pix_1.png" width = "30%">
   ##### cGAN loss
   > * The original cGAN (condition GAN) loss is described as below:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/cGAN.png" width = "30%">
   > * SmoothL1 loss:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/smoothL1.png" width = "30%">
   ##### reconsruction loss
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/reconstruction loss.png" width = "30%">
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lreconx_y.png" width = "30%">
   ##### decomposition loss
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/decomposition loss.png" width = "30%">
   > * The decompostion loss makes the image in different brightness is decomposed to the same illumination images. 
   
   ##### SSIM-MS loss
   > * And the enhancement loss optimize the L1 distance of composite image and target image which can be described as:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/enhancement loss.png" width = "30%">
   > * where *y* is the target image, and *Rx·Ix'* is the composite image.
   > * In order to obtain the image details, we use the better SSIMMS loss which are proposed by Zhao et al. "Loss functions for image restoration with neural networks", 2017. The SSIM-MS loss is a multi-scale version of SSIM loss which comes from the Structural SIMilarity index (SSIM). Means and standard deviations are computed with a Gaussian ﬁlter with standard deviation  , . SSIM for pixel p is deﬁned as:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/SSIM" width = "30%">
   > * where µi; µj is the average of i; j, σi^2; σj^2 is the variance of i; j,
             σij is the convariance of i and j, 
             c1 = (k1·L)^2; c2 = k2·L^2 are two variables to stabilize the division with weak denominator,
             L is the dynamic range of the pixel-values, 
             k1 = 0:01 and k2 = 0:03 by default. 
             The loss function for SSIM can be then written as:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lssim" width = "30%">
   > * while SSIM Loss are influenced by the parameters σG, Zhao et al. use the MS_SSIM rather than fine-tuning the σG, Given a dyadic pyramid of M levels, MS_SSIM is defined as:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/MS_SSIM" width = "30%">
   > * The multiscale SSIM loss for patch p is defined as:
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lssim_ms" width = "30%">
   > * Combine the Lenh with Lssim_ms and take the strategies by Zhao et al.
   > * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lcom" width = "30%">
   > * where α is set to 0.84.

## 4. Experimental Results & Discussions
### A. Converted See in the Dark Dataset
##### Qualitative Analysis
##### Quantitative Analysis

### B. LOL Dataset
##### Comparison of Decomposition Results
##### Comparison in the Real Scene

### C. Ablation Study on CSID
### D. Discussion

## 5. Conclusion


## Q & A 
**Q1: **
    * A:
