# Learning PyTorch Basic
## Convolutional Neural Network
#### Intro
* CNN is similar to ordinary neural networks and they are made up of neurons that have learned weights and biases.
* The main difference is that CNN mainly work on image data and apply these so-called convolutional filters so a typical CNN contains: ConV + activation function (e.g. ReLU) and followed by a Pooling layer.
* These layers are used to automatically learn some features from the images and then, at the end, we have one or more fully connected layers for actual classification tasks.
#### Layers
1. Convolutional layer
    * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/%60%7DG%25ICMTDH~P%7BU8%5B%5BJCAUMB.png" width="50%">
    * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/Y%7BSWVK4%241%5B%243DY(%24%5B%7B%25X(%5BI.png" width="50%">
2. Max pooling layer: used to down-sampling, reduce computational costs by reducing the size of the image so this reduces the number of parameters our model has to learn. And it helps to avoid overfitting by providing an abstracted form of the input.
    * <img src="https://raw.githubusercontent.com/TheLissandra1/Nest-of-Lisa/master/ImageLinks/Y%25T)K0CU7QVL3D%60~MM7W5%7BR.png" width="50%">
3. Fully-connected layer: 全连接层（fully connected layers，FC）在整个卷积神经网络中起到“分类器”的作用。如果说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空间的话，全连接层则起到将学到的“分布式特征表示”映射到样本标记空间的作用。在实际使用中，全连接层可由卷积操作实现：对前层是全连接的全连接层可以转化为卷积核为1x1的卷积；而前层是卷积层的全连接层可以转化为卷积核为hxw的全局卷积，h和w分别为前层卷积结果的高和宽.(注1）
   * 注1: 有关卷积操作“实现”全连接层，有必要多啰嗦几句。以VGG-16为例，对224x224x3的输入，最后一层卷积可得输出为7x7x512，如后层是一层含4096个神经元的FC，则可用卷积核为7x7x512x4096的全局卷积来实现这一全连接运算过程，其中该卷积核参数如下：“filter size = 7, padding = 0, stride = 1, D_in = 512, D_out = 4096”经过此卷积操作后可得输出为1x1x4096。如需再次叠加一个2048的FC，则可设定参数为“filter size = 1, padding = 0, stride = 1, D_in = 4096, D_out = 2048”的卷积层操作。
   * 链接：https://www.zhihu.com/question/41037974/answer/150522307
   * 消除spatial的影响
#### Code Example
*
```python 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__() # 不要忘了super
        self.conv1 = nn.Conv2d(3, 6, 5) # input channel is 3 because color img has 3 channels, and output_size is 6, kernel_size is 5, 5*5 kernel
        self.pool = nn.MaxPool2d(2, 2) # pooling layer: with kernel_size is 2 and stride is 2 步长
        self.conv2 = nn.Conv2d(6, 16, 5) # input_size必须和上一个输出一样都是6，这一层的output_size是16
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层，用于分类，输出120个feature，16*5*5是确定的
        self.fc2 = nn.Linear(120, 84) # 全连接层2， 输入120个feature，输出84个
        self.fc3 = nn.Linear(84, 10) # 84->10 feature，输出必须是10个，因为我们有10个classes

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 1205*5
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')           
           
```
* * * 
### 如果accuracy低，可以增加epoch值
