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
   > * Three objectives: 1) each pixel value of the enhanced image should be in the normalized range of [0,1] to avoid information loss induced by overflow truncation; 2) this curve should be monotonous to preserve the differences (contrast) of neighboring pixels; 3) the form of this curve should be as simple as possible and  differentiable in the process of gradient backpropagation.
   > * Curve: 
   >   <img src='', width=80%>
   > *
### B. Regularization Loss 💜

**Code Interpretation
```python

```

### C. Multitask Loss 💜 
##### 


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
