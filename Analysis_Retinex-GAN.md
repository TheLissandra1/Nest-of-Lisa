# Low-light Image Enhancement Algorithm Based on Retinex and Generative Adversarial Network
* * *
## 1. Intro
#### 1. 
   1.

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
  > 1. 

## 3. Proposed Research
### A. Network Structure  💜
   > * Based on Retinex theory.
   > * <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Retinex_Theory.png', width="80%">
   > * Unet: 
   >   <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Unet.jpg', width="80%">
   
### B. Regularization Loss 💜
   > * We assume that RGB channels of images are exposed to the same level of light (Retinex), thus we use the Unet to split the image S into reﬂected image R with three channels and illumination image I with one channel. It has proved that the assumption fails to maximize the network performance while illumination image I and reﬂected image R can’t reconstruct the original image S effectively (in a paper named 'Lightness and retinex theory' written by E. H. Land and J. J. McCann. in 1971). Then we try the second strategy. The author assume that the illumination image is also **three channels**. On the basis of this assumption, there is a serious problem that the network quickly falls into a local optimal solution. 
   > * Brief analysis of local optimal solution:
   > * Given 2 matrix S1 and S2, if we want to optimize min|R1-R2| to satisfy the following equation for all points i, j:
   > * <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimizeR1-R2.png', width="50%">
   > * The best two solutions are as below；
   > * <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimals solution1.png', width="50%">
   > * <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/optimals solution2.png', width="50%">
   > * We can see that after many iterations, the value of reﬂected image will become all 1 or -1 and the illumination image will be same as original or the reversed image. This means that the decomposition is meaningless. To solve this problem, the author propose a regularization loss *Lreg* which can prevent the RGB values of generated illumination image from approaching 1 or -1 to avoid that the network falls into a local optimal solution.
   > * <img src='https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_Retinex-GAN/Lregularization.png', width="70%">

* * *
### C. Multitask Loss 💜 
   > * 
##### 1. 


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
