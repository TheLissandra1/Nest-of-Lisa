## EnlightenGAN
### Training Possible Errors:
#### 1. IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
* Use .item() to fix it.
##### 1st training: batch size = 4
* It begins in 17:36 and ends in 01:00, lasts 7.5 hours, 200 epochs, the learning rate decreases from 0.0001 at 101th epoch and falls to 0.000001 in the final 200th epoch. Did not use the visdom server.
* Note: Only batch size <= 4 and with 3 GPU (0,1,2) is available. Batch size larger than 4 will cause Runtimeerror: CUDA out of memory.
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/training1.png" width="80%">
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/training3.png" width="80%">

##### later training attempts:
* Attempts to training only succeed once, the author uses ```python -m visdom.server``` to visualize in real time, but when trying to connect through remote server, it always report 2 kinds of errors:
1. Connection error:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/trainingConnectionError.png" width="70%">
* I have tested to connect the visdom through ssh with simple demos and it worked well. However, when it comes to EnlightenGAN, all attempts failed.
 * <img src = "https://raw.githubusercontent.com/TheLissandra1/Visdom_Connected_to_Remote_Server_Tutorial/main/Images/visdom.png" width="90%">
2. Out of Memory:
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/trainingMemoryError.png" width="80%">
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/Pytorch.png" width="90%">
* It seems that the Pytorch always occupys a large proportion of the memory, and I have tried many times in different GPU servers, including fortunato, usher, dupin and prospero. The out of memory error begins after loading unaligned dataset and at the beginning of initializing the network.
#### Testing Dataset Preview
* We choose standard ones used in previous works (NPE, LIME, MEF, DICM, VV, etc.).
* Noted: Fusion dataset image sizes are various, NPE dataset image formats and sizes are various.
*
| images | NPE | LIME | MEF | DICM | VV | LOL | Brightening | Fusion |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| detail | 750 * 725 pixel.jpg format | 720 * 680 pixel.bmp format | 512 * 340 pixel.png format | 480 * 640 pixel.JPG format | 2304 * 1728 pixel.jpg |600 * 400 pixel .png format |384 * 384 pixel .png | 900 * 573 .png |
|image e.g.| <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/birds.jpg"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/1.bmp"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/A.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/01.JPG"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/P1000205.jpg"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/2_real_A.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/r000da54ft_real_A.png"> | <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks_EnlightenGAN/Tests/1_real_A.png"> |
### Testing Possible Errors:
#### 1. If we use images from LIME dataset to test the pretrained model, the memory only allows to test 5 images in one time.
