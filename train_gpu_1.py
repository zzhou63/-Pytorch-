
"""
利用GPU 训练（一）
"""
# from model_standard import *
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="/Users/chikako/Desktop/pythonProject1/data",
                                          train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10(root="/Users/chikako/Desktop/pythonProject1/data",
                                          train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)


# 看训练集，和 测试集 有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#  利用 DataLoader 来加载数据集
train_dataloadr = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
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
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2 # 10的负二次方
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch= 10

# 添加tensorboard
writer = SummaryWriter("/Users/chikako/Desktop/pythonProject1/logs_train")

for i in range(epoch):
    print("---------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloadr:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 没有梯度，可以保证调优
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = ( outputs.argmax(1) == targets ).sum()
            total_accuracy = total_accuracy + accuracy


    print("整体测试集的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/ test_data_size))
    writer.add_scalar("test_loss", total_test_loss,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    # torch,save( tudui.state_dict(), "tudui_{}.pth".format(i) )
    print("模型已保存")

writer.close()

