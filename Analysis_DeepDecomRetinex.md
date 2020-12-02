# Deep Decomposition Retinex Based Low-light Image Enhancement
## Retinex-Net based low-light image enhancement
* The classic Retinex theory models the human color perception. It assumes that the observed images can be decomposed into two components, reflectance and illumination. Let *S* represent the source image, then it can be denoted by this formula, where *R* represents reflectance, *I* represents illumination and ◦ represents element-wise multiplication.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/SI.png">
* Reflectance describes the intrinsic property of captured objects, which is considered to be consistent under any lightness conditions. The illumination represents the various
lightness on objects. On low-light images, it usually suffers from darkness and unbalanced illumination distributions.
* * * 
## Pipeline
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/Retinex-Net.png" width="90%">

## Dataset Preview
*  **low light image: resolution: 600 * 400 pixel (width * height), created by Adobe Light-room, .png format**
* <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/lowlightimg2.png">
* * * 
*  **corresponding normal light image: resolution: 600 * 400 pixel (width * height), created by Adobe Light-room, .png format**
* <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/2.png">
```python
```
* * * 
### I. Decom-Net 
* It takes in pairs of low/normal-light images at the training stage, while only low-light images as input at the training at the testing stage.
* With the constraints that the low/normal-light images share the same reflectance and the smoothness of illumination, Decom-Net learns to extract the consistent *R* between variously illuminated images in a data-driven way.
* During training, there is no need to provide the ground truth (正确的标记) of the reflectance and illumination. Only requisite knowledge including the consistency of reflectance and the smoothness of illumination map is embedded into the network as loss functions. 
* Thus, the decomposition of our network is automatically learned from paired low/normal-light images, and by nature suitable for depicting the light variation among the images under different light conditions.
##### Decom-Net Code Interpretation
1. Decom-Net takes the low-light image *Slow* and the normal-light one *Snormal* as input, then estimates the reflectance *Rlow* and the illumination *Ilow* for *Slow*, as well as *Rnormal* and *Inormal* for *Snormal*, respectively. 
2. It first uses a 3 × 3 convolutional layer to extract features from the input image.
3. Then, several 3×3 convolutional layers with Rectified Linear Unit (ReLU) as the activation function are followed to map the RGB image into reflectance and illumination. A 3×3 convolutional layer projects *R* and *I* from feature space, and sigmoid function is used to constrain both *R* and *I* in the range of [0, 1].
* * * 
####    1. Conv + ReLU + Sigmoid
* 
```python


```
* * * 
####    2. Forward
* 
```python

```
* * *
#### Loss Functions
**1. The loss *L* consists of 3 terms: reconstruction loss *Lrecon*, invariable reflectance loss *Lir*, and illumination smoothness loss *Lis*:**
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/loss.png">
* 
### II.1 Enhance-Net
* The Enhance-Net takes an overall framework of encoder-decoder. A multi-scale concatenation is used to maintain the global consistency of illumination with context information in large regions while tuning the local distributions with focused attention.
* By mitigating the effect of total variation at the places where gradients are strong, the constraint successfully smooths the illumination map and retains the main structures.
* * * 
##### Enhance-Net Code Interpretation
* 
```python
```
* * * 
* 
```python
```
### II.2 Denoise on Reflectance
* The amplified noise, which often occurs in low-light conditions, is removed from reflectance if needed.
### III. Reconstruction
* We combine the adjusted illumination and reflectance by element-wise multiplication at the reconstruction stage.
### Training
* * * 
* * * 
### Prediction
* * * 
* * * 
### Evaluation
* * * 
* * * 
