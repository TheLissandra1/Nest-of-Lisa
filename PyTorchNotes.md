# Learning PyTorch Basic 
## Tensor Basics:
#### In Numpy arrays and vectors
#### BUT in PyTorch, everything is a Tensor, so a tensor can have different dimensions, e.g. 1d, 2d, or even 3d or more.
#### Calculation
  ```python
  import torch
      x = torch.empty(3)
      print(x)
  ```
  * ` torch.empty(3)`创建了一个一维向量，包含三个elements
  * ` torch.empty(2,3)` 创建了一个2行3列的二维矩阵
  * ` torch.zeros(), torch.ones()`
  * ` x = torch.ones(2,2,dytpe = torch.float16)` 定义类型
  * ` x.size()` 查看dimension
  * ` torch.tensor([2.5, 0.1]) ` 创建list
  * All PyTorch functions with a '_' tail will do an in-place operation.
  ```python
  x = torch.rand(2,2)
  y = torch.rand(2,2)
  z = x - y equals  to z = torch.sub(x,y)
  ```    
  * Multiply ` torch.mul()`  Sustitute: `torch.mul_()`
  * Division ` torch.div()`
* * * 
#### Slice operation
 ```python
 x = torch.rand(5,3)
 print(x)
 print(x[;, 0])
 ```
  * We can see the first column in all rows with :,0
  * We can see certain element e.g.` x[1,1]`
  * Use `.item() `to show the whole value
  * ` x.view() ` To reshape
  * If use `x.view(-1, 8)` ,by enter the first parameter = -1, PyTorch will help determine the write dimension with 8 columns.
  * * * 
#### From Numpy to torch
###### Numpy run in CPU and Torch run in GPU, so we cannot transform a tensor working on GPU back to Numpy
  ```python
    a = np.ones(5)  
    b = torch.from_numpy(a, dtype = float16) 
  ```
  * Use this to transport a numpy array into a tensor
  * ` a += 1` Increment in each value
  ```python
  if torch.cuda.is_available():
       device = torch.device("cuda") # cuda means GPU
       x = torch.ones(5, device=device) # set x runs in GPU
       y = torch.ones(5) # originally y runs in CPU
       y = t.to(device) # set y to run in GPU
       z = x + y # this will run in GPU and it will be faster
       z = z.to("cpu") # this will run in CPU again 
   ```
* * * 
## Gradient Calculation with Autograd  
  * ` x = torch.ones(5, requires_grad=True) `
  * ` torch.randn()` 生成随机standard normal distribution的 [0, 1]之间的数
  * 
    ```python
    x = torch.randn(3, requires_grad=True)
    
    y = x + 2
    z = y*y*2
    
    z = backward() # dz/dx 
    print(x.grad)
    # In background, it creates a so-called vector Jacobian product to get gradients
    ```
    * <img src = "https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/G%40JRP3U0X_E474H(%5D_E%24KXH.png">
    * This left one is Jacobian matrix, the middle one is the gradient vector, and the right one is the final gradients that we are interested in. This is the so-called 'Chain rule'.
    * `backward(v)` requires input as a vector if input has an argument. However, with no input `backward()` can accept a scalar
    #### How to prevent from tracking the gradients
      * Three options
        1. ` call `x.requires_grad_(False)`
        2. ` x.detach() ` # this will create a new vector/tensor with the same values, but it dpesn't require the gradient
        3. ```python
            with torch.no_grad():
              y = x + 2
              print(y)
           ```
     * Whenever we call the backward function, the gradient for this tensor will be accumulated into the dot grad attribute, so their values will be summed up.
     ##### Example
     ```python
     weights = torch.ones(4, requires_grad=True)
     
     for epoch in range(2): # 2 iteration
           model_output = (weights*3).sum()
           model_output.backward()
           print(weighs.grad)
         # If we run the code now, gradients will be accumulated as the sum()
          
         # before we do the next iteration and optimization step, we must empty the gradients
         # so we need to do this:
         weights.grad.zero_()
         # now if we run this, our gradients are correct
     ```
     ###### However, we can simply use PyTorch built-in optimizer to do this (a simple example)
     ```python
     weights = torch.ones(4, requires_grad=True)
     
     optimizer = torch.optim.SGD(weights, lr=0.01)
     optimizer.step()
     optimizer.zero_grad()
     
     ```
     
              
              
     
       
     
