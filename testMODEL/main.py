import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from testMODEL.AlexNet import AlexNet
from testMODEL.trainFunc import train_and_save
from testMODEL.testFunc import test
from testMODEL.loadModel import load_model


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 初始化模型并移动到设备
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练并保存模型
train_and_save(model, train_loader, criterion, optimizer, num_epochs=10, save_path='./alexnet_mnist.pth')

# 测试模型
test(model, test_loader)

# 继续训练模型
load_model(model, './alexnet_mnist.pth')
train_and_save(model, train_loader, criterion, optimizer, num_epochs=5, save_path='./alexnet_mnist.pth')

# 再次测试模型
test(model, test_loader)