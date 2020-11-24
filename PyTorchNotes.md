# Learning PyTorch Basic 
## Tensor Basics:
### In Numpy arrays and vectors
### BUT in PyTorch, everything is a Tensor, so a tensor can have different dimensions, e.g. 1d, 2d, or even 3d or more.  
  ` import torch
      x = torch.empty(3)
      print(x)`
  * ` torch.empty(3)`创建了一个一维向量，包含三个elements
  * ` torch.empty(2,3)` 创建了一个2行3列的二维矩阵
  * ` torch.zeros(), torch.ones()`
  * ` x = torch.ones(2,2,dytpe = torch.float16)` 定义类型
  * ` x.size()` 查看dimension
  * ` torch.tensor([2.5, 0.1]) ` 创建list
  * All PyTorch functions with a '_' tail will do an in-place operation.
  ` x = torch.rand(2,2)
    y = torch.rand(2,2)
    z = x*y
    equals ` to `z = torch.sub(x,y)`
  * 
