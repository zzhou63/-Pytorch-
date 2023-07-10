"""
网络模型的保存及读取
"""
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

vgg16 = torchvision.models.vgg16(pretrained=False) # 未经过训练的参数，我们初始化了一个参数
# 保存方式一: 保存了结构和参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2: 保存了参数(官方推荐), 内存更小
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")

