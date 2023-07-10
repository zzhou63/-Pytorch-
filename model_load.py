"""
网络模型的保存及读取
"""
from model_save import *  # 这样写 就不用 复制 class，反正都在一个文件里
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 方式1 -》 保存方式1: 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False) # 未经过训练的参数，我们初始化了一个参数
model1 = torch.load("/Users/chikako/Desktop/pythonProject1/vgg16_method1.pth")
# print(model1)

# 方式2 -》 保存方式2: 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False) # define the structure of nn
vgg16.load_state_dict(torch.load("/Users/chikako/Desktop/pythonProject1/vgg16_method2.pth"))
# model2 = torch.load("/Users/chikako/Desktop/pythonProject1/vgg16_method2.pth") # get the dictory in py
# print(vgg16)

# 通常模式，只得到 参数，没有结构
# model2 = torch.load("/Users/chikako/Desktop/pythonProject1/vgg16_method2.pth") # get the dictory in py
# print （model2）

# 若要加上 结构，如下操作
vgg16 = torchvision.models.vgg16(pretrained=False)  # define the structure of nn
vgg16.load_state_dict(torch.load("/Users/chikako/Desktop/pythonProject1/vgg16_method2.pth"))
# print(vgg16)

# 陷阱2
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui,self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
#
# tudui = Tudui() 不需要写这一步
# torch.save(tudui, "tudui_method1.pth")
model = torch.load('tudui_method1.pth')
print(model)