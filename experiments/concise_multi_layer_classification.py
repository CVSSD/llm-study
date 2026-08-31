"""
concise_multi_layer_classification.py
与 multiLayer_classification.py 结构完全一致的多层感知机（PyTorch 简洁版）：
  Linear(784,100) -> ReLU -> Linear(100,10)
  SGD(lr=0.01) + L2 weight decay 0.001（仅权重，不含偏置）
  batch_size=128, num_epoch=30, seed=42
用于与手写版对比结果。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('IS GPU available:', torch.cuda.is_available())

transform = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])

batch_size = 128
lr = 0.01
num_epoch = 30
weight_decay = 0.001

train_set = datasets.FashionMNIST(root='/home/yan/proj/data', train=True, download=True,
                                  transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = datasets.FashionMNIST(root='/home/yan/proj/data', train=False, download=True,
                                 transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)   # 与手写版 w1(784,100), b1(100) 对应
        self.fc2 = nn.Linear(100, 10)    # 与手写版 w2(100,10), b2(10) 对应
        # 与手写版初始化完全一致：w ~ N(0, 0.01), b = 0
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)        # (N,784)
        x = F.relu(self.fc1(x))          # 训练/验证统一用 ReLU
        return self.fc2(x)               # logits


net = MLP().to(device)

# 手写版只对 w1/w2 加 L2，不加到偏置 -> 用参数分组精确复刻
optimizer = optim.SGD([
    {'params': net.fc1.weight, 'weight_decay': weight_decay},
    {'params': net.fc1.bias},
    {'params': net.fc2.weight, 'weight_decay': weight_decay},
    {'params': net.fc2.bias},
], lr=lr)

train_loss, test_loss, test_acc = [], [], []
for epoch in range(num_epoch):
    net.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.view(-1, 28 * 28).to(device), y.to(device)

        optimizer.zero_grad()
        loss = F.cross_entropy(net(X), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    train_loss.append(running_loss / len(train_loader.dataset))

    # 验证：与训练同一套前向（net.eval() 关 dropout/batchnorm，这里没有也保持习惯）
    net.eval()
    with torch.no_grad():
        running_loss = 0
        correct = total = 0
        for X, y in test_loader:
            X, y = X.view(-1, 28 * 28).to(device), y.to(device)

            logits = net(X)
            running_loss += F.cross_entropy(logits, y).item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        test_loss.append(running_loss / len(test_loader.dataset))
        test_acc.append(correct / total)

    print(f'epoch {epoch + 1}: train_loss={train_loss[-1]:.4f} '
          f'test_loss={test_loss[-1]:.4f} test_acc={test_acc[-1]:.3f}')

print(f'\n=== concise final ===  test_loss={test_loss[-1]:.4f}  test_acc={test_acc[-1]:.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_loss, label="Train"); ax1.plot(test_loss, label="Test")
ax1.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="Loss")
ax1.legend(); ax1.grid(True)
ax2.plot(test_acc, label="Test Acc", color='tab:green')
ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig('concise_mlp_curves.png')
plt.show()
