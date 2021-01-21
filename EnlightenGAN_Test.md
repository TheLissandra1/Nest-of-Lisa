## EnlightenGAN
### Training Possible Errors:
#### 1. IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
* Use .item() to fix it.
#### 1st training: batch size = 4
* begins in 17:36
* ends in 01:00, lasts 7.5 hours, 200 epochs, the learning rate decreases from 0.0001 at 101th epoch and falls to 0.000001 in the final 200th epoch.
* Note: Only batch size <= 4 and with 3 GPU (0,1,2) is available. Batch size larger than 4 will cause Runtimeerror: CUDA out of memory.

### Testing Possible Errors:
#### 1. If we use images from LIME dataset to test the pretrained model, the memory only allows to test 5 images in one time.
