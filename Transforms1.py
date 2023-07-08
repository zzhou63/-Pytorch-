"""
在 terminal 里面输入tensorboard --logdir=logs
前往图片所在地

快捷键使用方法： Ctrl+ P, 查看需要的数据类型
Transform 的使用

tensor 的数据类型
通过 transform.ToTensor 去解决两个问题
1. transform该如何使用
2. 为什么需要tensor数据类型
"""
from PIL  import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# 为什么需要tensor数据类型
# 从 transforms中选择一个class，进行一个创建
# 利用创建的工具得到我们想要的结果
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()