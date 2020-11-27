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

