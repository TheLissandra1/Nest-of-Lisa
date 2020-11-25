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
##### Pipeline:
1. So we loop over the epochs again and then we do another loop and loop over all the batches,
2. And then we get the X and Y batch samples,
3. And do the optimization based only on those batches
4. PyTorch can do batch calculations and iterations for us, so it's easy to use.
### Some Terms about batch training
|Term | Explanation |
| :-----| :----------|
| epoch | 1 forward and backward pass of all training samples |
| batch_size | number of training samples in one forward and backward pass |
| number of iterations | number of passes, each pass using [batch_size] number of samples |
##### E.g. 100 samples, batch_size = 20--> 100/20 = 5 iterations for 1 epoch
