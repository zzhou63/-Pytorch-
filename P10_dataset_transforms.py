"""
torchvision中的数据集的使用方式
"""

import torchvision
from torch.utils.tensorboard import SummaryWriter

datasets_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

train_set = torchvision.datasets.CIFAR10(root="./datasets",train=True,transform=datasets_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasets",train=False,transform=datasets_transform,download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10): # 0 - 9
    img, target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()

