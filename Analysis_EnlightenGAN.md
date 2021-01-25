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
 > * Perceptual loss: https://arxiv.org/pdf/1603.08155.pdf
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
* We first compare the visual quality of EnlightenGAN with several recent competing methods. The results are demonstrated in Fig. 4, where the first column shows the original low-light images, and the second to fifth columns are the images enhanced by: a vanilla CycleGAN trained using our unpaired training set, RetinexNet, SRIE, LIME, and NPE. The last column shows the results produced by EnlightenGAN.
* We next zoom in on some details in the bounding boxes. LIME easily leads to over-exposure artifacts, which makes the results distorted and glaring with the some information missing. The results of SRIE and NPE are generally darker compared with others. CycleGAN and RetinexNet generate unsatisfactory visual results in terms of both brightness and naturalness. In contrast, EnlightenGAN successfully not only learns to enhance the dark area but also preserves the texture details and avoids over-exposure artifacts.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Fig4.png" width="110%">
* Figure 4: Comparison with other state-of-the-art methods. Zoom-in regions are used to illustrate the visual difference. 
    - First example: EnlightenGAN successfully suppresses the noise in black sky and produces the best visible details of yellow wall.
    - Second example: NPE and SRIE fail to enhance the background details. LIME introduces over-exposure on the woman’s face. However, EnlightenGAN not only restores the background details but also avoids over-exposure artifacts, distinctly outperforming other methods. 
    - Third example: EnlightenGAN produces a visually pleasing result while avoiding overexposure artifacts in the car and cloud. Others either do not enhance dark details enough or generate over-exposure artifacts. Please zoom in to see the details.
#### 4.3.2 Human Subjective Evaluation 
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/comparison.png" width="110%">

#### 4.3.3 No-Referenced Image Quality Assessment
* We adopt Natural Image Quality Evaluator (NIQE) https://live.ece.utexas.edu/publications/2013/mittal2013.pdf, a well-known no-reference image quality assessment for evaluating real image restoration without ground-truth, to provide quantitative comparisons. The NIQE results on five publicly available image sets used by previous works (MEF, NPE, LIME, VV, and DICM) are reported in Table 1: a lower NIQE value indicates better visual quality. EnlightenGAN wins on three out of five sets, and is the best in terms of overall averaged NIQE. This further endorses the superiority of EnlightenGAN over current state-of-the-art methods in generating high-quality visual results.
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
* Figure 6: Visual comparison of the results on the BBD-100k dataset. EnlightenGAN-N is the domain-adapted version of EnlightenGAN, which generates the most visually pleasing results with noise suppressed. Please zoom in to see the details.

### 4.5 Pre-Processing for Improving Classification
* Image enhancement as pre-processing for improving subsequent high-level vision tasks has recently received increasing attention, with a number of benchmarking efforts. We investigate the impact of light enhancement on the extremely dark (ExDark) dataset, which was specifically built for the task of low-light image recognition. The classification results after light enhancement could be treated as an indirect measure on semantic information preservation, as suggested.
* The ExDark dataset consists of 7,363 low-light images, including 3000 images in training set, 1800 images in validation set and 2563 images in testing set, annotated into 12 object classes. We use its testing set only, applying our pretrained EnlightenGAN as a pre-processing step, followed by passing through another ImageNet-pretrained ResNet-50 classifier. Neither domain adaption nor joint training is per-formed. The high-level task performance serves as a fixed semantic-aware metric for enhancement results.
* In the low-light testing set, using EnlightenGAN as pre-processing improves the classification accuracy from 22.02% (top-1) and 39.46% (top-5), to 23.94% (top-1) and
40.92% (top-5) after enhancement. That supplies a side evidence that EnlightenGAN preserves semantic details, in addition to producing visually pleasing results. We also conduct experiment using LIME and AHE. LIME improves the accuracy to 23.32% (top-1) and 40.60% (top-5), while AHE obtains to 23.04% (top-1) and 40.37% (top-5).
## Conclusion
* In this paper, we address the low-light enhancement problem with a novel and flexible unsupervised framework.
* The proposed EnlightenGAN operates and generalizes well without any paired training data. The experimental results on various low light datasets show that our approach outperforms multiple state-of-the-art approaches under both subjective and objective metrics.
* Furthermore, we demonstrate that EnlightenGAN can be easily adapted on real noisy lowlight images and yields visually pleasing enhanced images.
* Our future work will explore how to control and adjust the light enhancement levels based on user inputs in one unified model. Due to the complicacy of light enhancement, we also expect integrate algorithm with sensor innovations.





## Code
### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

#### parameters setting: scripts/script.py
```python
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot ../final_dataset \
		--no_dropout \
		--name enlightening \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 256 \
        --patchSize 32 \
		--skip 1 \
		--batchSize 2\
        --self_attention \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0,3" \
	    "--display_port=" + opt.port)

elif opt.predict:
	for i in range(1):
	        os.system("python predict.py \
	        	--dataroot ../test_dataset \
	        	--name enlightening \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
                --self_attention \
                --times_residual \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(200 - i*5))
```

#### train.py
```python
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

opt = TrainOptions().parse()
config = get_config(opt.config)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data() # dataset [UnaligneDataset] was created
dataset_size = len(data_loader) # 1016
print('#training images = %d' % dataset_size) # #training images = 1016

model = create_model(opt) # single
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1): # (1, 201=100+100+1), 200 iterations
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize # origin value = 32, defined in script.py
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data) # input
        model.optimize_parameters(epoch)

        if total_steps % opt.display_freq == 0:  # total_steps % 30 (frequency of showing training results on screen)
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:   # total_steps % 100 (frequency of showing training results on console)
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0: # 1, True
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0: # total_steps % 5000
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0: # epoch % 5
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    # End of epoch %d / 200
    # Time Taken: %d sec
    if opt.new_lr: # True
        if epoch == opt.niter: # epoch == 100
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):  # epoch == 120
            model.update_learning_rate()
        elif epoch == (opt.niter + 70): # epoch == 170
            model.update_learning_rate()
        elif epoch == (opt.niter + 90): # epoch == 190
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()

```

#### Dataloader
```python
class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A') # --dataroot ../final_dataset/trainA
        # self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B') # --dataroot ../final_dataset/trainB

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A) # function defined in image_folder.py
        # dir_A = "../final_dataset/trainA"
        # A_imgs: all images stored in trainA folder, A_paths: all images paths stored in trainA folder
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths) # A_size = 914
        self.B_size = len(self.B_paths) # B_size = 1016
        
        self.transform = get_transform(opt) # function defined in base_dataset.py
                                            # randomly crops images into opt.fineSize
                                            # then get normalized to [-1,1]

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        A_img = self.transform(A_img) # get_transform, randomly crop and normalization
        B_img = self.transform(B_img)

        
        if self.opt.resize_or_crop == 'no': # False, resize_or_crop = 'crop'
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else: # True
            w = A_img.size(2) # width
            h = A_img.size(1) # height
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else: # True
                input_img = A_img
            if self.opt.lighten:# True, normalize attention map. ????
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img)) # min-max normalization
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. # Gray = R*0.299 + G*0.587 + B*0.114?????
            A_gray = torch.unsqueeze(A_gray, 0) # dimension = 0
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

```
#### model.py
* 
```python
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'pix2pix')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pair':
        # assert(opt.dataset_mode == 'pair')
        # from .pair_model import PairModel
        from .Unet_L1 import PairModel
        model = PairModel()
    elif opt.model == 'single': # use this single model by default in script.py
        # assert(opt.dataset_mode == 'unaligned')
        from .single_model import SingleModel
        model = SingleModel()
    elif opt.model == 'temp':
        # assert(opt.dataset_mode == 'unaligned')
        from .temp_model import TempModel
        model = TempModel()
    elif opt.model == 'UNIT':
        assert(opt.dataset_mode == 'unaligned')
        from .unit_model import UNITModel
        model = UNITModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

```
#### single_model.py: EnlightenGAN model initialize
```python
class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size) # CUDA: out of memory
        # nb: batchSize, opt.input_nc: 3, input image channels, size: fineSize (randomly cropped image size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        # opt.output_nc: 3, output image channels
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0: # vgg = 1 in scripts.py, use vgg by default
            self.vgg_loss = networks.PerceptualLoss(opt) # Self feature preserving loss
            if self.opt.IN_vgg: # True, patch vgg individual
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids) # load pretrained vgg16 model
            self.vgg.eval() # switch to evaluation mode
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0: # use semantic loss
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False # skip = 1
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, # netG_A is G: Unet_resize_conv
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        # self.netG_A = networks.define_G(3,3,64,sid_unet_resize,instance, not 'no dropout for the generator',gpu_ids, skip=1,opt)
        # ngf: num of gen filters in first conv layer, 64 by default
        # skip: = 1 in scripts.py. B = net.forward(A) + skip*A
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain: # True when training
            use_sigmoid = opt.no_lsgan # do *not* use least square GAN
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, # discriminator (global)
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            # self.netD_A = networks.define_D(3,64,NoNormDiscriminator,5,instance,no sigmoid,gpu_ids,patch=False)
            if self.opt.patchD: # True, use patch discriminator (local)
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
                # self.netD_P = networks.define_D(3,64,NoNormDiscriminator,4,instance,no sigmoid,gpu_ids,patch=True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain: # True when training
            self.old_lr = opt.lr # lr = 0.0001, defined in train_options.py
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)# 50, the size of image buffer that stores previously generated images
            # define loss functions
            if opt.use_wgan: # False
                self.criterionGAN = networks.DiscLossWGANGP()
            else: # True
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse: # True
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))# beta1=0.5, momentum of adam
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD: # True
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A) # sid_unet_resize aka Unet_resize_conv
        # networks.print_network(self.netG_B)
        if self.isTrain: # True
            networks.print_network(self.netD_A)  # no_norm_4
            if self.opt.patchD: # True
                networks.print_network(self.netD_P) # use patch discriminator
            # networks.print_network(self.netD_B)
        if opt.isTrain:# True
            self.netG_A.train()
            # self.netG_B.train()
        else: # predict mode
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

```

##### single_model.py: backward() of D_A (global), D_P (local) and G (generator)
```python
############################################backward_D_basic()#########################################	
    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real) # D(real)
        pred_fake = netD.forward(fake.detach()) # D(fake)
        if self.opt.use_wgan: # False
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan: # True,
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D
############################################backward_D_A()#########################################	
    def backward_D_A(self): # Global
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
############################################backward_D_P()#########################################	    
    def backward_D_P(self): # Local
        if self.opt.hybrid_loss: # True
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0: # True
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2: # True
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()
############################################backward_G()#########################################	
    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan: # True
            pred_real = self.netD_A.forward(self.real_B)

            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        loss_G_A = 0
        if self.opt.patchD: # True
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss: # True
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0: # 5 True
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])
                    
                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2
                    
            if not self.opt.D_P_times2: # False
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)
            else: # True
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A*2
                
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0: # True vgg = 1
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                else: # True
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0: # 5, True
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else: # True
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w
       
        self.loss_G.backward()
############################################optimize_parameters#########################################
    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else: # True
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()

```
#### networks.py: generator (G) and discriminator (D)
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/EnlightenGAN_model_G_Unet_resize_conv.png" width="100%">
```python

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, skip=skip, opt=opt)
    elif which_model_netG == 'unet_512':
        netG = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, skip=skip, opt=opt)
    elif which_model_netG == 'sid_unet':
        netG = Unet(opt, skip)
    elif which_model_netG == 'sid_unet_shuffle':
        netG = Unet_pixelshuffle(opt, skip)
    elif which_model_netG == 'sid_unet_resize': # use this model in script.py
        netG = Unet_resize_conv(opt, skip)
    elif which_model_netG == 'DnCNN':
        netG = DnCNN(opt, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm_4': # use this model in script.py
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'no_patchgan':
        netD = FCDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, patch=patch)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD
    
    
    
    #######################################################################
    # Generator (G)
    class Unet_resize_conv(nn.Module): # revoke this in G
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        if opt.self_attention:
            self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            # self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        else:
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size*block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).resize(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
            if self.opt.self_attention:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
                # x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x*gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
            
            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent*gray

            # output = self.depth_to_space(conv10, 2)
            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    output = latent + input*self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output/torch.max(torch.abs(output))
            
        ######        
        # In script.py we use norm = 1 so I delete the code of elif self.opt.use_norm == 0:# 
                                                                                     #######
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output

    #######################################################################
    # Discriminator (D)
    class NoNormDiscriminator(nn.Module): # revoke this in D in script.py
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


```
#### networks.py: Vgg16
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/VGG16_D.jpg"> width = "70%">
```python
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2) 
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3

```
#### Perceptual Loss, GAN loss (LSGAN loss, MSE loss)
```python
class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance: # action='store_true' in base_options.py, help='vgg instance normalization'
            return torch.mean((img_fea - target_fea) ** 2) # perceptual loss
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
```
* * * 
```python
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label # 1
        self.fake_label = target_fake_label # 0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss() # MSELoss=(Xi-Yi)^2, return loss.mean(), a scalar
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real: # True
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))# numel() returns elements number
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else: # False
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

```
## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.


### Testing process

Download [pretrained model](https://drive.google.com/file/d/1AkV-n2MdyfuZTFvcon8Z4leyVb0i7x63/view?usp=sharing) and put it into `./checkpoints/enlightening`

Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict```
#### predict.py
```python
import time
import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()

```
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







