# Learning PyTorch Basic
## Softmax and Cross Entropy
#### Softmax 
* Softmax Formula: It does normalization by applying the exponential function to each element and then divide by the sum of all exponentials, it squashes the output to be between 0 and 1 and get the probabilities.
* <img src="https://github.com/TheLissandra1/Nest-of-Lisa/blob/master/ImageLinks/%247GWMZDEZLW6F3~~)T3%24VM9.png" width="30%">
* There are 1 linear layer which has 3 output values are so-called raw values (scores/logits) at first, then we apply softmax and get probabilities, so each value is sqaushed between 0 and 1. The highest value gets the highest probability and the sum of probabilities is 1.
* The Y_pred is the prediction value, it will choose for the class with the highest probability.
* Softmax Layer  <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/_H8KW285LGCYNXO2JMCY%5BVI.png" width="50%">
#### Code Example
* 
```python
import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0) # implementation of Softmax function
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0) # dim=0 compute along the 1st axis
print(outputs)

```
#### Cross-Entropy: Softmax is often combined with cross-entropy loss
* Cross-Entropy: It measures the performance of classification model whose output is a probability between 0 & 1, and it can be used in multi class problems
* And the loss increases as the predicted probability diverges from the actual label, so the better our prediction the lower is our loss so here we have 2 examples.
* So here we have a good prediction and a low cross-entropy loss and below is a bad prediction and a high cross-entropy loss.
* The raw values classes are ONE-HOT codding
* <img src="https://github.com/TheLissandra1/Nest-of-Lisa/blob/master/ImageLinks/0%24M%600RHA)%5BM%5DW%7BRGEE%5BWKVP.png" width="60%">
#### Code Example with Numpy
*
```python
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted)) # implementation as cross-entropy formula in graph above
    return loss
# y must be one hot encoded
# if class 0: [1 0 0] # 0, 1, 2 rows occupied by 1 and others are 0
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

# y_pred has probabilities
Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

```
* * *
#### Code Example with PyTorch
##### Careful! nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss), therefore, no Softmax in the last layer. And, Y has class labels, not One-Hot, Y_pred has raw scores (logits), not Softmax
* <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/2b48b7a0ed3a94b641929a5da16928e.png" width = "60%">
*
```python
import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples * nclasses = 1*3
Y_pred_good = torch.tensor([2.0,1.0,0.1]) # raw values
Y_pred_bad = torch.tensor([])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item()) # only have one value so we can call item() 
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
```
* * *
* <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/afda4798204f78b2fdcfe1fb09da4db.png">
* Convert to a binary problem
<img src="https://github.com/TheLissandra1/Nest-of-Lisa/blob/master/ImageLinks/cd2324db059211e2a532ced9b1a7d74.png" width="50&">
