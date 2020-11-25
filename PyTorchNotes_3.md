# Learning PyTorch Basic_Part 3
##  Dataset and DataLoader-Batch Training
#### Code
* 
```python
data = numpy.loadtxt('wine.csv')
# training loop
for epoch in range(1000):
    x, y = data
    # forward + backward + weight updates
```
* If we load the data like above, and looped over the number of epochs and then optimized our model based on the whole data set,
* This could be very time consuming if we did gradient calculations on training data
* SO, a better for large data sets is to divide the samples into smaller batches, 
* Then our training loop should be like this
```python
# training loop 
for epoch in range(1000):
    # loop over all batches
    for i in range(total_batches):
        x_batch, y_batch = ...
# --> use DataSet and DataLoader to load wine.csv    
```
#### Pipeline:
1. So we loop over the epochs again and then we do another loop and loop over all the batches,
2. And then we get the X and Y batch samples,
3. And do the optimization based only on those batches
4. PyTorch can do batch calculations and iterations for us, so it's easy to use.
#### Some Terms about batch training
|Term | Explanation |
| :-----| :----------|
| epoch | 1 forward and backward pass of all training samples |
| batch_size | number of training samples in one forward and backward pass |
| number of iterations | number of passes, each pass using [batch_size] number of samples |
##### E.g. 100 samples, batch_size = 20--> 100/20 = 5 iterations for 1 epoch
### Code Example with PyTorch
*
```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading 
        xy = np.loadtxt('./data/wine/wine.csv', detimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:]) # all samples without 1st column
        self.yy = torch.from_numpy(xy[:, [0]]) # all n_samples but only 1st column
        self.n_samples = xy.shape[0] # the 1st dimension is the number of samples
        
    def __getitem__(self,index):
        # dataset[0]
        return self.x[index], self.y[index] # this will return a tuple
    def __len__(self):
        # len(dataset)
        return self.n_samples
        
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# See how to use dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2) # shuffle means random

datatiter = iter(dataloader)
data = datatiter.next()
features, labels = data
print(features, labels)

################
# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)  # ceil 向大取整, 4 is the batch_size
print(total_samples, n_iterations) # 178 45


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader): # enumerate(), 枚举，返回一个索引序列
        # forward and backward pass, update weights
        if (i+1) % 5 ==0: # every 5 step
            print(f'epoch{epoch+1}/{num_epochs},step{i+1}/{n_iterations},inputs{inputs.shape}')
           
```
* * *
## Dataset Transforms
#### transforms 
* [documentation: ] (https://pytorch.org/docs/stable/torchvision/transforms.html)
*
| On Images | On Tensors | Conversion | Generic | Custom | Compose multiple Transforms |
| :-------| :-------- | :----- | :------- | :------- | :------- |
| CenterCrop, Grayscale, Pad, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomRotation, Resize, Scale | LinearTransformation, Normalize, RandomErasing | ToPILImage: from tensor or ndarray; To Tensor: from numpy.ndarray or PILImage | Use Lambda | Write own class | composed = transforms.Compose([Rescale(256), RandomCrop(224)])   torch.vision.transforms.ReScale(256)   torchvision.transforms.ToTensor() |


