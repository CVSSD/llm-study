import time
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Is GPU available: {torch.cuda.is_available()}')


transform = torchvision.transforms.v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128

train_set = torchvision.datasets.CIFAR10(root='/home/yan/proj/data', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='/home/yan/proj/data', train=False,
                                        download=True, transform=transform)
test_load = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.manual_seed(42)

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Cnn().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

train_loss, test_loss, test_acc, epoch_time = [], [], [], []
for epoch in range(2):
    start = time.time()
    running_loss = 0.0
    num_samples = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    train_loss.append(running_loss / num_samples)

    # 验证集：只前向，算 loss + accuracy
    net.eval()
    with torch.no_grad():
        running_loss = 0
        correct = total = 0
        for i, data in enumerate(test_load):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        test_loss.append(running_loss / total)
        test_acc.append(correct / total)

    elapsed = time.time() - start
    epoch_time.append(elapsed)
    print(f'epoch {epoch + 1}: train_loss={train_loss[-1]:.4f} '
          f'test_loss={test_loss[-1]:.4f} test_acc={test_acc[-1]:.4f} '
          f'time={elapsed:.1f}s')

print('Training finish')
print(f'avg epoch time: {sum(epoch_time) / len(epoch_time):.1f}s')
