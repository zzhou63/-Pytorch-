"""
损失函数
反向传播操作
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
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    result_loss = loss(output,targets)
    # 反向传播
    result_loss.backward()

