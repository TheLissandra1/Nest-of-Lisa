# Learning PyTorch Basic__Part2
### Prediction, gradients computation, loss computation, parameter updates
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
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([2],[4],[6],[8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}') # call item() because we want to see the value rather than tensor
    
# Training
learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss() # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
        
    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 10 ==0: # %10
        [w, b] = model.parameters()
        print(f'epoch{epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f} ')
 print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
  
  ```
