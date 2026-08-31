import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Is GPU available {torch.cuda.is_available()}')

batch_size = 128
lr = 0.01
lambd = 0.01

transform = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

train_set = torchvision.datasets.FashionMNIST(root='/home/yan/proj/data', train=True, download=True,
                                              transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.FashionMNIST(root='/home/yan/proj/data', train=False, download=True,
                                              transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

torch.manual_seed(42)
w = (torch.randn(28 * 28, 10, dtype=torch.float32) * 0.01).to(device).requires_grad_()
b = torch.zeros(10).to(device).requires_grad_()

train_loss, test_loss, test_acc, regulization = [], [], [], []
for epoch in range(15):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        X, y = data
        X, y = X.view(-1, 28 * 28).to(device), y.to(device)

        loss = F.cross_entropy(X @ w + b, y)
        running_loss += loss.item() * X.size(0)
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    train_loss.append(running_loss / len(train_loader.dataset))
    regulization.append((w ** 2).sum().item())

    with torch.no_grad():
        running_loss = 0
        correct = total = 0
        for i, data in enumerate(test_loader, 0):
            X, y = data
            X, y = X.view(-1, 28 * 28).to(device), y.to(device)

            y_hat = X @ w + b
            loss = F.cross_entropy(y_hat, y)
            running_loss += loss.item() * X.size(0)

            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        test_loss.append(running_loss / len(test_loader.dataset))
        test_acc.append(correct / total)

    print('epoch finish')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
ax1.plot(train_loss, label="Train"); ax1.plot(test_loss, label="Test")
ax1.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="Loss")
ax1.legend(); ax1.grid(True)
ax2.plot(test_acc, label="Test Acc", color='tab:green')
ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
ax2.legend(); ax2.grid(True)
ax3.plot(regulization, label='REG')
ax2.set(xlabel="Epoch", ylabel="REG", title="REG")
ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.show()
