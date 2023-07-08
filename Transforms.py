"""
Transform 的使用

tensor 的数据类型
通过 transform.ToTensor 去解决两个问题
1. transform该如何使用
2. 为什么需要tensor数据类型
"""
from PIL  import Image
from torchvision import transforms


img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

# 1. transform该如何使用
# 从 transforms中选择一个class，进行一个创建
# 利用创建的工具得到我们想要的结果
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)