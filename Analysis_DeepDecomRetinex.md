# Deep Decomposition Retinex Based Low-light Image Enhancement
## Retinex-Net based low-light image enhancement
* The classic Retinex theory models the human color perception. It assumes that the observed images can be decomposed into two components, reflectance and illumination. Let *S* represent the source image, then it can be denoted by this formula, where *R* represents reflectance, *I* represents illumination and ◦ represents element-wise multiplication.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/SI.png">
* **Q：What is element-wise multiplication?**
* **A：In mathematics, the Hadamard product (also known as the element-wise, entrywise or Schur product) is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands, where each element i, j is the product of elements i, j of the original two matrices.**
*  <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/element-wise%20multiplication.png" width="50%">
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
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # input image channel number == 4, output channels produced by the convolution, convolving kernel size = 3
        # zero-padding added to both sides of the input == 4, padding_mode = 'replicate'

        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        # Activated layers!
        #
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')


    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

```
* * * 
####    2. Forward
* 
```python

```
* * *
#### Loss Functions
**1. The loss *L* consists of 3 terms: reconstruction loss *Lrecon*, invariable reflectance loss *Lir*, and illumination smoothness loss *Lis*:**
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/loss.png" width='80%'>
* where *lir* and *lis* denote the coefficients to balance the consistency of reflectance and the smoothness of illumination.

* **1.1 The *Lrecon* is defined as:**
* This is the regularization which prevent the model from doing too well on training data.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/lossRecon.png" width='90%'>
* Based on the assumption that both *Rlow* and *Rhigh* can reconstruct the image with the corresponding illumination map, the reconstruction loss *Lrecon* is formulated as above.
* The formula means that *Lrecon* equals to the sum of (coefficients of every pixel muliply the L1 Norm of *Ri* element-wise multiply *Ij* minus *Sj*), where *i* and *j* are low and normal index.
* Subscripts in *i = low, normal* means this *lis* calculation formula works on both low and normal light images.
* L1 Norm: the sum of absolute values of differences.
* **Q: Why we use L1 Norm here?**
  **A: To sparse the weights, thus we can complete feature selection and add model interpretability. And if compared with L2 norm, L1 create less features and minimize the weights much faster than L2; L1 is also Robust to abnormal values.
* And, if we rethink about *Ri* in *Lrecon* after viewing *Lir*, we know that Reflectance of low and normal images are same due to constraints, so we don't need to care too much about *Ri* here.
* Therefore, if the input is:
  1. low light image, then *Lrecon* = ∑∑ λij*||Reflectance o Illumination of low image- low image||1.
  2. normal light image, then *Lrecon* = ∑∑ λij*||Reflectance o Illumination of normal image- normal image||1.
* ```python
  # paste some code here
  ```
* **1.2 Invariable reflectance loss *Lir* is introduced to constrain the consistency of reflectance:**
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/LossInvariableReflectance.png" width="70%">
* **My comments: The author mentioned that low light and normal images should share the same reflectance in any conditions because reflectance is the intrinsic property of objects. However, there might be color differences between normal and low light conditions. And that's why we need to minimize this constraint *Lir* to ensure image pairs have same Reflectance before the next step--Enhance-Net.**


* **1.3 The *Lis* is defined as:**
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/LossIlluminationSmoothness.png" width="80%">
* where *∇* denotes the gradient including *∇h (horizontal)* and *∇v (vertical)*, and *lg* denotes the coefficient balancing the strength of structure-awareness. With the weight *exp(−lg∇Ri)*, *Lis* loosens the constraint for smoothness where the gradient of reflectance is steep, in other words, where image structures locate and where the illumination should be discontinuous. 
* Subscripts in *i = low, normal* means this *lis* calculation formula works on both low and normal light images.
### II.1 Enhance-Net
#### Background Theory
* One basic assumption for illumination map is the local consistency and the structure- awareness. In other words, a good solution for illumination map should be smooth in textural details while can still preserve the overall structure boundary.
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
