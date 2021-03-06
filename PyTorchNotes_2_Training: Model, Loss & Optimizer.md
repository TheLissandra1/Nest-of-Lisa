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

model = nn.Linear(input_size, output_size) # This is simple and only 1 layer

# Try a custom model########################
##This is the same as the argument in line 27 above
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        retrun self.lin(x)
    model = LinearRegression(input_size, output_size)
#############

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
  
  
* * *
## Linear Regression
  *
  ```python
  import torch 
  import torch.nn as nn
  import numpy as np
  from sklearn import datasets 
  import matplotlib.pyplot as plt
  # 0. prepare data
  X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1,noise=20, random_state=1)
  # convert to a torch tensor, before the data is double, we need transform the data to float
  X = torch.from_numpy(X_numpy.astype(np.float32))
  y = torch.from_numpy(y_numpy.astype(np.float32))
  # make a column vector, reshape the tensor
  y = y.view(y.shape[0],1)
  
  n_samples, n_features = X.shape  
  # 1. model
  input_size = n_features
  output_size = 1
  model = nn.Linear(input_size, output_size)
  
  # 2. loss and optimizer
  learning_rate = 0.01
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
  # 3. training loop
  num_epochs = 100
  
  for epoch in range(num_epochs):
      # forward pass and loss
      y_predicted = model(X)
      loss = criterion(y_predicted, y)
      # backward pass
      loss.backward()
      
      # update
      optimizer.step()
      # empty the gradients
      optimizer.zero_grad()
      if (epoch+1)%10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
  
  # plot 
  predictd = model(X).detach().numpy() # convert it from a new tensor to numpy # use detach() so the gradients attribute is set to false
  plt.plot(X_numpy, y_numpy, 'ro')
  plt.plot(X_numpy, predicted, 'b')
  plt.show()
  ```
  
  * * *
  ## Logistic Regression
  *
    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler # to scale our features
    from sklearn.model_selection import train_test_split
    
    # 0. prepare data
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    
    n_samples, n_features = X.shape
    print(n_samples, n_features) # 569, 30
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    # scale our features
    sc = StandardScaler() # recommended to do in logistic regression, which will make features have zero mean and unit variance 
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test) # Perform standardization by centering and scaling
    
    X_train = torch.from_numpy(X_train.astype(np.float32)
    X_test = torch.from_numpy(X_test.astype(np.float32)
    y_train = torch.from_numpy(y_train.astype(np.float32)
    y_test = torch.from_numpy(y_test.astype(np.float32)
    
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)
    
    
    
    
    
    # 1. model
    # f = wx + b, sigmoid at the end
    class LogisticRegression(nn.Module):
        def __init__(self, n_input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)
        def forward(self,x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted
    model = LogisticRegression(n_features)
            
    # 2. loss and optimizer
    learning_rate = 0.01
    criterion = nn.BCELoss() # binary cross-entropy loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 3. training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)
        # backward pass
        loss.backward()
        # updates
        optimizer.step()
        # zero gradients
        optimizer.zero_grad()
        if (epoch+1) % 10 == 0:
            print(f'epoch:{epoch+1}, loss = {loss.item():.4f}')
     
     with torch.no_grad(): # In PyTorch, it is a context manager, 被with torch.no_grad() wrap起来的语句，将不会track gradients
     # evaluation 
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0]) # shape[0] returns the number of elements in y_test
        # 大写命名X表示二位矩阵，小写y表示一维向量
        print(f'accuracy = {acc:.4f}')
        
        print()
    
    ```
