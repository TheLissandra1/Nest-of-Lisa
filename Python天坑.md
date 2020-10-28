### 1. pip安装/更新一直不成功，用下面的命令试试，重点是--user

  pip install --user --upgrade pip
  

### 2. anaconda prompt 创建环境

  问题：

  [WinError 126] 找不到指定的模块

  解决：

- [Q1] 原因是anaconda3/DLLs中缺少libcrypto-1_1-x64.dll，anaconda3\Library\bin中缺少libcrypto-1_1-x64.dll。在这两个路径下查看这个两个dll文件是否被加了后缀或者被更改了命名。              anaconda3\Library\bin下的libcrypto-1_1-x64.dll应该和anaconda3/DLLs中的libcrypto-1_1-x64.dll日期保持一致。

- [Q2] 原因是anaconda3/DLLs中缺少libssl-1_1-x64.dll，anaconda3\Library\bin中缺少libssl-1_1-x64.dll。在这两个路径下查看这个两个dll文件是否被加了后缀或者被更改了命名。anaconda3\Library\bin下的libssl-1_1-x64.dll应该和anaconda3/DLLs中的libssl-1_1-x64.dll日期保持一致。


### 3. anaconda切换python版本

  使用conda create -n <environment name> python=<version number> 命令创建环境，指定python版本
   - [Q1] e.g.  conda create -n python38 python=3.8

### 4. 如何查看anaconda安装了什么环境

    conda env list
    
    - [Q1]使用环境： conda activate <环境名>
### 5. anaconda python38环境下pip install显示  Could not fetch URL https://pypi.python.org/simple/: connection error: HTTPSConnectionPool(host='pyp

   错误： ①Could not fetch URL https://pypi.python.org/simple/: connection error: HTTPSConnectionPool(host='pyp ② Could not find a version that satisfies the requirement scikit-learn==0.19.1 (from versions: ) 
   
   解决：使用国内镜像并trust
   
  - [Q1]  e.g. pip install numpy -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
  
 ### 6. pycharm 提示" package requirement opencv-contrib-python is not satisfied "
 
    解决： pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python
    重点是加 --user
    
    
