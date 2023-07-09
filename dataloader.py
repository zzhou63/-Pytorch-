"""
DataLoader的使用
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的数据集
test_data = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=torchvision.transforms.ToTensor())

# num_workers 进程数, =0 是指只有一个主进程
# drop_last = True，最后余下的部分会被舍去，不会保留余数
test_loader = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

# 测试数据集中第一张图片及target, target 是标签存放的位置，真正的标签是 classees[target]
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs,step)
        step = step + 1

writer.close()

"""
torch.Size([4, 3, 32, 32]) 4 张图片，3通道， 32*32
tensor([1, 6, 3, 9])      把4个target进行打包
"""
