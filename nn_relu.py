"""
神经网络 - 非线形激活
RELU: 小于0 的 数返回0， 不小于0的数 返回原值
非线性越多，模型泛化能力越强
"""
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1,3]],dtype=torch.float32)

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data", train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs,step)
    output = tudui(imgs)
    writer.add_images("output", output,step)
    step = step + 1

writer.close()

