# Learning PyTorch Basic__Part2
## Training Pipeline: Model, Loss and Optimizer
### Steps
  1. Design model (input size, output size, forward pass) # design input and output size and design all different forward pass with all the different operations or between all the different layers
  2. Construct loss and optimizer
  3. Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
#### Example with PyTorch (Modified from the last example in previous notes)
  * 
  ```python
  import torch
  import torch.nn as nn
# f = w*x
# f = 2*x
X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)
w = torch.tensor(0.0, dtype = torch.float32,requires_grad=True)
# model prediction
def forward(x):
    return w*x
    

# Training
learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss() # mean squared error
optimizer = torch.optim.SGD(w, lr=learning_rate) # stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
        
    # zero gradients
    optimizer.zero_grad()
    
    if epoch %1 ==0: # %1 means ONE step size
        print(f'epoch{epoch+1}: w = {w:.3f}, loss = {l:.8f} ')
 print(f'Prediction after training: f(5) = {forward(5):.3f}')
  
  ```
