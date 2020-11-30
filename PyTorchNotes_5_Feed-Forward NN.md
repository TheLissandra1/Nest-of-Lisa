# Learning PyTorch Basic
## Feed-Forward Neural Network
#### Pipeline
1. MNIST dataset (example)
2. DataLoader, Transformation
3. Multilayer Neural Net, activation function
4. Loss and Optimizer
5. Training Loop (batch training)
6. Model evaluation
7. GPU support (if you have one)
#### Code Example
* 
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28*28
hidden_size = 100
num_classes =10 # because we have digits from 0 to 9
num_epochs = 2 # this is small so training won't take too long
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', # we store the data in a root path and convert dataset to a tensor
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True) 
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())         

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
                                          
# let's look at one batch of the data
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)
# torch.Size([100, 1, 28, 28]) torch.Size([100])
# 100 is the batch_size and 1 is one channel (no color channels here), 28, 28 is actual our image array (according to img size 28*28, labels is a tensor, each class has a label value.

# have a look on the dataset
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray') # cmap is color map, it shows grayscale img
# plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # 1st layer
        self.relu = nn.ReLU() # activation function
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out) # process the previous output
        out = self.l2(out)
        return out 
 
 model = NeuralNet(input_size, hidden_size, num_classes)
 
 # loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # training loop
  n_total_steps = len(train_loader)
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):
      # shape: 100, 1, 28, 28
      # input_size: 784, (100,784) 2d tensor
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)
      
      # forward
      outputs = model(images)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if (i+1) % 100 == 0:
          print(f'epoch{epoch+1}/num_epoche, step{i+1}/n_total_steps), loss={loss.item():.4f}')
    
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # value, index
        _, predictions = torch.max(outputs, 1)
        

```
