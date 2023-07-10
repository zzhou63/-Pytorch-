"""
优化器 + 模型训练
"""

from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

dataset = torchvision.datasets.CIFAR10("/Users/chikako/Desktop/pythonProject1/data", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

# 模型训练， 实际训练中 循环次数成百上千甚至上万
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        # 得到调用的梯度
        result_loss = loss(output,targets)
        # 反向传播
        # result_loss.backward()

        # 优化器
        optim.zero_grad() # 每个参数对应的梯度清零
        result_loss.backward() # get grad
        optim.step() #对这个参数调优
        running_loss = running_loss + result_loss # 整体误差的求和
    print(running_loss)
