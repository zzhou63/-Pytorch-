# 在terminal里面输入： tensorboard --logdir=logs
# 可以打开tensor
# 如果担心指示端重复， 则可以自己命名， 如下：
#  tensorboard --logdir=logs --port=6007， 6007 是自己规定的地址，可按情况修改

# 命令快捷键：
# 查看函数使用，长按 ctrl，然后点击函数名
# 在terminal中 Ctrl+ C， 是退出


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

 # writer.add_image()
# y = 2x

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
