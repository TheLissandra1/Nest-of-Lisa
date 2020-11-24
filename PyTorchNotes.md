# Learning PyTorch Basic 
## Tensor Basics:
### In Numpy arrays and vectors
### BUT in PyTorch, everything is a Tensor, so a tensor can have different dimensions, e.g. 1d, 2d, or even 3d or more.
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
  z = x - y equals ` to `z = torch.sub(x,y)
  ```    
  * Multiply ` torch.mul()`  Substitute: `torch.mul_()`
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
   ` a = np.ones(5)  
     b = torch.from_numpy(a, dtype = float16) `
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
  * ` x = torch.ones(5, requires_grad=True)`
  
     
       
     
