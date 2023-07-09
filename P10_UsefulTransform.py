from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_image/0013035.jpg")
print(img)

# ToTensor
# create an object, 实例对象化
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# 然后img可以放入tensorboard 当中
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0, 3, 2],[6, 2, 9] )
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512 , 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor ->img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - Resize
"""
Compose 中 的参数需要是一个列表
Python中， 列表的表示形式为【数据1， 数据2，。。。】
在Compose中， 数据需要是 transforms 类型，
所以得到， Compose（【transforms参数1， transforms参数2，。。。】）
"""
trans_resize_2 = transforms.Resize(512)

# PIL img -> PIL img -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((500, 700))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
