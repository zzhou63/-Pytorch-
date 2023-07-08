# 在terminal里面输入： tensorboard --logdir=logs
# 可以打开tensor
# 如果担心指示端重复， 则可以自己命名， 如下：
#  tensorboard --logdir=logs --port=6007， 6007 是自己规定的地址，可按情况修改

# 命令快捷键：
# 查看函数使用，长按 ctrl，然后点击函数名
# 在terminal中 Ctrl+ C， 是退出

# Use Opencv 读取图片， 获得numpy type 图片

""" 将 'PIL.JpegImagePlugin.JpegImageFile'> 转化成 numpy型
import numpy as np
img_array = np.array(img)
print(type(img_array))
<class 'numpy.ndarray'>
"""

"""
从PIL到 numpy， 需要在add_image()中指定shape中每一个数字/维 表示的含义
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)

# writer.add_image("test", img_array, 2, dataformats='HWC')
"""如果想要不同的title显示，则每一次重新命名即可"""
writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
