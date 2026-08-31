import torch
import torchvision
import random
import matplotlib.pyplot as plt
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('IS GPU available:', torch.cuda.is_available())

transform = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

batch_size = 128
lr = 0.01
num_epoch = 30
lambd = 0.001

train_set = torchvision.datasets.FashionMNIST(root='/home/yan/proj/data', train=True,
                                              download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.FashionMNIST(root='/home/yan/proj/data', train=False,
                                             download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

torch.manual_seed(42)
w1 = (torch.randn(28 * 28, 100, dtype=torch.float32) * 0.01).to(device).requires_grad_()
b1 = torch.zeros(100).to(device).requires_grad_()
w2 = (torch.randn(100, 10, dtype=torch.float32) * 0.01).to(device).requires_grad_()
b2 = torch.zeros(10).to(device).requires_grad_()

def ReLu(y):
    return torch.clamp(y, min=0)

def Sigmoid(y):
    return 1 / (1 + torch.exp(-y))

def tanh(y):
    return (torch.exp(2 * y) - 1) / (1 + torch.exp(2 * y))

train_loss, test_loss, test_acc = [], [], []
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.view(-1, 28 * 28).to(device), y.to(device)

        y_hat_1 = ReLu(X @ w1 + b1)
        y_hat = y_hat_1 @ w2 + b2

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        running_loss += loss.item() * X.size(0)
        loss.backward()

        with torch.no_grad():
            w1 -= lr * (w1.grad + lambd * w1)
            b1 -= lr * b1.grad
            w2 -= lr * (w2.grad + lambd * w2)
            b2 -= lr * b2.grad
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()

    train_loss.append(running_loss / len(train_loader.dataset))

    with torch.no_grad():
        running_loss = 0
        correct = total = 0
        for i, data in enumerate(test_loader):
            X,y = data
            X, y = X.view(-1, 28 * 28).to(device), y.to(device)

            y_hat_1 = ReLu(X @ w1 + b1)
            y_hat = y_hat_1 @ w2 + b2
            
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item() * X.size(0)

            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        test_loss.append(running_loss / len(test_loader.dataset))
        test_acc.append(correct / total)

    print('Epoch finish')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_loss, label="Train"); ax1.plot(test_loss, label="Test")
ax1.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="Loss")
ax1.legend(); ax1.grid(True)
ax2.plot(test_acc, label="Test Acc", color='tab:green')
ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
ax2.legend(); ax2.grid(True)
#ax3.plot(regulization, label='REG')
#ax2.set(xlabel="Epoch", ylabel="REG", title="REG")
#ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.show()
