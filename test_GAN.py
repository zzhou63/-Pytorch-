"""
完整的模型验证套路
"""
import torch as torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from PIL import Image

img_path= "./imgs/airplane.png"

image = Image.open(img_path)
print(image)

image = image.convert('RGB')

transform =torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32 ,5 ,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32 ,32 ,5 ,1 ,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32 ,64 ,5 ,1 ,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4 ,64),
            nn.Linear(64 ,10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model1 = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))
print(model1)
image = torch.reshape(image, (1,3,32,32))
model1.eval()
with torch.no_grad():
    output = model1(image)
print(output)

print(output.argmax(1))