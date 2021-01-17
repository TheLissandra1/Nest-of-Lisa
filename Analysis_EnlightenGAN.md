# EnlightenGAN: Deep Light Enhancement without Paired Supervision
## Intro
* Proposed a highly effective unsupervised generative adversarial network, dubbed **EnlightenGAN**, that can be trained without low/normal-light image pairs, yet proves to generalize very well on various real-world test images.
* **Q：What is element-wise multiplication?**
* **A：In mathematics, the Hadamard product (also known as the element-wise, entrywise or Schur product) is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands, where each element i, j is the product of elements i, j of the original two matrices.**
*  <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/element-wise%20multiplication.png" width="50%">
* Reflectance describes the intrinsic property of captured objects, which is considered to be consistent under any lightness conditions. The illumination represents the various
lightness on objects. On low-light images, it usually suffers from darkness and unbalanced illumination distributions.
* * * 
## Pipeline
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/Retinex-Net.png" width="90%">

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
####    1. Conv + ReLU
* 
```python
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate') #  4-->64
        # Activated layers
        ## kernel size = 3*3 is a common size, the smaller the kernel size is, the smaller the computational complexity is
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), # 64-->64
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), # 64-->64
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), # 64-->64
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), # 64-->64
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'), # 64-->64
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate') # 64-->4

```
####    2. Forward in DecomNet
* 
```python
    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0] # the max value in the column direction of the input image
        # select the max values (illumination) among RGB 3 channels and concatenate them to 4 channels
        # in order to match the input channels in the 1st conv layer and the last convolutional layer 'recon_layer'
        input_img= torch.cat((input_max, input_im), dim=1) # concatenate input max and input image in column direction
        feats0   = self.net1_conv0(input_img) # 96--96 64 channel
        featss   = self.net1_convs(feats0) # 96--96 64 channel # output_size =  (input_size-filter_size+2*padding)/Stride+1
        outs     = self.net1_recon(featss) # 96--96 4 channel
        R        = torch.sigmoid(outs[:, 0:3, :, :]) # put img R into [0,1]  3 channel Reflectance
        L        = torch.sigmoid(outs[:, 3:4, :, :]) # put img I into [0,1]  1 channel Illumination
        return R, L # Return R (reflectance) and L(illumination)
```
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
  # Loss functions (all defined in RetinexNet in code)
  ```python
      def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda() # tranform input image from np.array to tensor datatype
        input_high= Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()
        R_low, I_low   = self.DecomNet(input_low) # obtain low Reflectance and low Illumination
        R_high, I_high = self.DecomNet(input_high) # obtain normal Reflectance and normal Illumination

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low) # obtain adjusted Illumination

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1) # build 3 channel Ilow, wherer I represents illumination
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1) # build 3 channel Inormal
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1) # build 3 channel Iadjusted
        # herr we create 3 channel variables to enable compute L1 norm with input 3 channel image in the next step
        
        # Compute losses
        # Define Lrecon in DecomNet
        self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low) # L1 norm (mean element-wise absolute value difference) between Rlow x Ilow3 and input low
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high) 
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low) 
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high) 
        
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach()) # Define Lir = L1 norm of (Rlow - Rnormal), Lir: invariable reflectance loss
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high) # Define Lrecon = L1 norm of (Rlow*I^-Snormal), Lrecon: thr reconstruction loss in RelightNet (EnhanceNet)
        
        # Define Lis in DecomNet and RelightNet (EnhanceNet), Lis: illumination smoothness loss, smooth() is defined by author
        self.Ismooth_loss_low   = self.smooth(I_low, R_low) # Lis in DecomNet
        self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low) # Lis in RelightNet
        
        # Loss of DecomNet = Lrecon + λir*Lir +  λis*Lis
        # λir = 0.001, λis = 0.1, λg = 10
        # when i ≠ j, λij = 0.001; when i = j, λij = 1
        self.loss_Decom = 1 * self.recon_loss_low + \
                          1 * self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \ 
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
                          # ? In paper the author states that λir = 0.001 but in code he set λir = 0.01 ?
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu() # Reflectance map of low light image, output of DecomNet
        self.output_I_low   = I_low_3.detach().cpu() # Illumination map of low light image, output of DecomNet
        self.output_I_delta = I_delta_3.detach().cpu() # adjusted illumination Idelta, output of RelightNet (EnhanceNet)
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu() # S is the final enhanced low-light image 

  
      # calculate gradient horizontally (along x) and vertically (along y)
      def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))                                   
        return grad_out
        
      def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)
                            

      def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

  ```


### II.1 Enhance-Net
#### Background Theory
* One basic assumption for illumination map is the local consistency and the structure- awareness. In other words, a good solution for illumination map should be smooth in 
textural details while can still preserve the overall structure boundary.
* The Enhance-Net takes an overall framework of encoder-decoder. A multi-scale concatenation is used to maintain the global consistency of illumination with context information 
in large regions while tuning the local distributions with focused attention.
* By mitigating the effect of total variation at the places where gradients are strong, the constraint successfully smooths the illumination map and retains the main structures.
* * * 
#### Loss function in RelightNet (EnhanceNet)
* The loss function *L* for Enhance-Net consists of the reconstruction loss *Lrecon* and the llumination smoothness loss *Lis*. *Lrecon* means to produce a normal-light *Sˆ*, 
which is,
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/Lrecon_Multi-Scale%20Illumination%20Adjustment.png" width="80%">
* *Rlow* element-wise multiply the adjusted *I^* equals to the reconstructed image *Srecon*, and by minimizing the difference between reconstructed image (it could be seen as a predicted image) and Normal light image *S*, thus we minimize the loss and get better performances.
* *Lis* is the same as <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/LossIlluminationSmoothness.png" width="70%"> , 
except that *Iˆ* is weighted by gradient map of *Rlow*
* * *
# EnhanceNet (it is called RelightNet in code)
```
#### Enhance-Net (RelightNet) Code Interpretation
####    1. ConV + DeConV + fusion
*  
```python
class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate') # 4-->64 kernel channels
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate') # 64
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate') # 64
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate') # 64
        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate') # 128
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate') # 256
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate') # 512
        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=1, padding_mode='replicate') # fuse 512*3=1536  use 1x1 conv to simply increase channel from 512 to 1536
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0) # 1536-->1 use 3x3 kernel to reconstruct illumination map I'

```
* * * 
####    2. Forward in RelightNet
*
```python
    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1) # concatenate R and L in dimension 1 (column direction) to enable 4 channels as input
        # 3 times convolution to implement down-sampling by setting stride = 2
        out0      = self.net2_conv0_1(input_img) # size: 96-->96
        # Why ReLU? 1. Reduce computational complexity; 2. Faster; 3. Sparse the network; 4. No Saturation.
        out1      = self.relu(self.net2_conv1_1(out0)) # 96-->48
        out2      = self.relu(self.net2_conv1_2(out1)) # 48-->24
        out3      = self.relu(self.net2_conv1_3(out2)) # 24-->12
        # Nearest Neighbour Interpolation to resize, thus avoid checkerboard artifacts
        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))  # resize the out3 to the given size: column 3 and 4 of out2: 12-->24
        # 3 times interpolation to implement up-sampling
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1))) # ReLU does not change image size, devonv1_1: 24+24-->48
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3])) # resize 48-->48
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1))) # 48+48-->96
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3])) # resize 96-->96
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1))) # 96+96--192
        
        # Multi-Scale features fusion to recover the Reflectance compoennt in multi-scale
        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3])) # resize 48-->96
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3])) # resize 96--96
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1) # 96+96+192-->384???

        feats_fus = self.net2_fusion(feats_all) # 96-1+2+1=98
        output    = self.net2_output(feats_fus) # 98-->96 1 channel
        return output # return adjusted Illumination component

```
### II.2 Denoise on Reflectance
* The amplified noise, which often occurs in low-light conditions, is removed from reflectance if needed.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_DeepDecomRetinex/Denoise.png" width="80%">
* However, there is no code implementation of Denoising on Reflectance at all, and I view the tensorflow version and there is no code implementation of denoising either. So I think maybe the author believes his model does not need this later on.

### Training
* * * 
* 
```python
    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(), # Adam is an optimizer can converge quickly and adaptive in learning rate adjustment.
                                           lr=lr[0], betas=(0.9, 0.999)) # and Adam is also the most popular optimizer in practice.
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high= np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img= Image.open(train_high_data_names[image_id])
                    train_high_img= np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _        = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img= np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img= np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img= np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img= np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :]= train_high_img
                    self.input_low = batch_input_low
                    self.input_high= batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)
```
* * * 
### Prediction
* * * 
*
```python
    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint

        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path  = test_low_data_names[idx]
            test_img_name  = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img   = Image.open(test_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            if save_R_L:
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image= np.concatenate([input, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir + '/' + test_img_name
            im.save(filepath[:-4] + '.jpg')
```
* * * 
### Evaluation
* * * 
* **Code example**

```Python
    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                       (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')
```
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







