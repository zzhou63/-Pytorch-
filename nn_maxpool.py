"""
神经网络 - 最大池化的使用
Example:视频转换
1080p -> 720p
"""
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype=torch.float32)
# # -1 表示模糊形状， 由torch自行计算
# input = torch.reshape(input, (-1, 1, 5,5))
# print(input.shape)

# 神经网络搭建
# 在池化核中， stride = kernel_size
# ceil_mode = True ：   采用ceil，保留余下的数中的最大值
# ceil_mode = False ：  不保留，根本不保存
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs,step)
    output = tudui(imgs)
    writer.add_images("output", output,step)
    step = step + 1

writer.close()



