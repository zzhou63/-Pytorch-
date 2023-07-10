"""
现有网络模型的使用及修改
"""
# 在 terminal 输入pip list 可查看现有的package
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

vgg16_false = models.vgg16(pretrained=False)
vgg16_true = models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("/Users/chikako/Desktop/pythonProject1/data",train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 开始炼丹
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)