import torch
import torchvision
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('IS GPU available:', torch.cuda.is_available())

transform = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

batch_size = 128
lr = 0.01
num_epoch = 100
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
w2 = (torch.randn(100, 100, dtype=torch.float32) * 0.01).to(device).requires_grad_()
b2 = torch.zeros(100).to(device).requires_grad_()
w3 = (torch.randn(100, 10, dtype=torch.float32) * 0.01).to(device).requires_grad_()
b3 = torch.zeros(10).to(device).requires_grad_()

weights = [w1, w2, w3]
weight_names = ['w1', 'w2', 'w3']
near_zero_threshold = 1e-6

def ReLu(y):
    return torch.clamp(y, min=0)

def Sigmoid(y):
    return 1 / (1 + torch.exp(-y))

def tanh(y):
    return (torch.exp(2 * y) - 1) / (1 + torch.exp(2 * y))

train_loss, test_loss, train_acc, test_acc = [], [], [], []
sample_every = 25
grad_history = []
gradient_ratio_history = []
weight_zero_count_history = []
weight_zero_ratio_history = []
persistent_zero_count_per_epoch = []
persistent_zero_ratio_per_epoch = []
persistent_zero_count_global = []
persistent_zero_ratio_global = []
global_persistent_zero_masks = None
for epoch in range(num_epoch):
    start_time = time.time()
    running_loss = 0.0
    correct = total = 0
    epoch_grads = []
    epoch_gradient_ratios = []
    epoch_zero_counts = []
    epoch_zero_ratios = []
    epoch_persistent_zero_masks = None
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.view(-1, 28 * 28).to(device), y.to(device)

        y_hat_1 = ReLu(X @ w1 + b1)         ; y_hat_1.retain_grad()
        y_hat_2 = ReLu(y_hat_1 @ w2 + b2)   ; y_hat_2.retain_grad()
        y_hat = y_hat_2 @ w3 + b3

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        running_loss += loss.item() * X.size(0)
        loss.backward()

        if i % sample_every == 0:
            g2 = y_hat_2.grad.norm().item()
            g1 = y_hat_1.grad.norm().item()
            epoch_grads.append([g2, g1])
            # Ratio in the backward direction: y_hat_2 -> y_hat_1.
            epoch_gradient_ratios.append([g1 / max(g2, 1e-12)])

            current_zero_masks = [
                weight.detach().abs() <= near_zero_threshold for weight in weights
            ]
            epoch_zero_counts.append([
                mask.sum().item() for mask in current_zero_masks
            ])
            epoch_zero_ratios.append([
                mask.float().mean().item() for mask in current_zero_masks
            ])

            if epoch_persistent_zero_masks is None:
                epoch_persistent_zero_masks = [mask.clone() for mask in current_zero_masks]
            else:
                epoch_persistent_zero_masks = [
                    persistent & current
                    for persistent, current in zip(epoch_persistent_zero_masks,
                                                   current_zero_masks)
                ]

            if global_persistent_zero_masks is None:
                global_persistent_zero_masks = [mask.clone() for mask in current_zero_masks]
            else:
                global_persistent_zero_masks = [
                    persistent & current
                    for persistent, current in zip(global_persistent_zero_masks,
                                                   current_zero_masks)
                ]

        with torch.no_grad():
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)

            w1 -= lr * (w1.grad + lambd * w1)
            b1 -= lr * b1.grad
            w2 -= lr * (w2.grad + lambd * w2)
            b2 -= lr * b2.grad
            w3 -= lr * (w3.grad + lambd * w3)
            b3 -= lr * b3.grad
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()
            w3.grad.zero_()
            b3.grad.zero_()

    grad_history.append(epoch_grads)
    gradient_ratio_history.append(epoch_gradient_ratios)
    weight_zero_count_history.append(epoch_zero_counts)
    weight_zero_ratio_history.append(epoch_zero_ratios)
    persistent_zero_count_per_epoch.append([
        mask.sum().item() for mask in epoch_persistent_zero_masks
    ])
    persistent_zero_ratio_per_epoch.append([
        mask.float().mean().item() for mask in epoch_persistent_zero_masks
    ])
    persistent_zero_count_global.append([
        mask.sum().item() for mask in global_persistent_zero_masks
    ])
    persistent_zero_ratio_global.append([
        mask.float().mean().item() for mask in global_persistent_zero_masks
    ])
    train_loss.append(running_loss / len(train_loader.dataset))
    train_acc.append(correct / total)

    with torch.no_grad():
        running_loss = 0
        correct = total = 0
        for i, data in enumerate(test_loader):
            X,y = data
            X, y = X.view(-1, 28 * 28).to(device), y.to(device)

            y_hat_1 = ReLu(X @ w1 + b1)
            y_hat_2 = ReLu(y_hat_1 @ w2 + b2)
            y_hat = y_hat_2 @ w3 + b3
            
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item() * X.size(0)

            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        test_loss.append(running_loss / len(test_loader.dataset))
        test_acc.append(correct / total)
    end_time = time.time()
    print(f'Epoch finish Time:{end_time - start_time:.3f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_loss, label="Train"); ax1.plot(test_loss, label="Test")
ax1.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="Loss")
ax1.legend(); ax1.grid(True)
ax2.plot(train_acc, label="Train Acc")
ax2.plot(test_acc, label="Test Acc", color='tab:green')
ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
ax2.legend(); ax2.grid(True)
#ax3.plot(regulization, label='REG')
#ax2.set(xlabel="Epoch", ylabel="REG", title="REG")
#ax2.legend(); ax2.grid(True)
plt.tight_layout()
# 先保存，再显示（show 在交互后端会阻塞/清空画布）
from pathlib import Path
save_dir = Path(__file__).resolve().parent / 'figs'
save_dir.mkdir(exist_ok=True)
fig.savefig(save_dir / '3layer_mlp.png', dpi=150, bbox_inches='tight')

# Gradient norms from the deepest hidden activation toward the input.
layer_x = [2, 1]
layer_labels = ['y_hat_2', 'y_hat_1']
target_epochs = [0, 29, 44, 59, 70, 99]

fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, ep in zip(axes, target_epochs):
    samples = grad_history[ep]
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(samples)))
    for k, sample in enumerate(samples):
        ax.plot(layer_x, np.log10(np.clip(sample, 1e-12, None)),
                marker='o', color=colors[k], alpha=0.7, linewidth=1.2)

    ax.set_title(f'Epoch {ep + 1}')
    ax.set_xticks(layer_x)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel('Layers')
    ax.set_ylabel(r'$\log_{10}(|gradient|)$')
    ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(save_dir / '3layer_grad_vs_layer.png', dpi=150, bbox_inches='tight')

# Near-zero weight ratio at the same sampled batches as the gradient statistics.
fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
weight_x = np.arange(len(weight_names))
for ax, ep in zip(axes, target_epochs):
    samples = weight_zero_ratio_history[ep]
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(samples)))
    for k, sample in enumerate(samples):
        ax.plot(weight_x, sample, marker='o', color=colors[k], alpha=0.7)
    ax.set_title(f'Epoch {ep + 1}')
    ax.set_xticks(weight_x)
    ax.set_xticklabels(weight_names)
    ax.set_xlabel('Weight matrices')
    ax.set_ylabel(f'Ratio with |w| <= {near_zero_threshold:g}')
    ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(save_dir / '3layer_near_zero_weights.png', dpi=150,
             bbox_inches='tight')

# Weights that remain near zero at every sampled batch.
fig4, (ax_epoch, ax_global) = plt.subplots(1, 2, figsize=(12, 4))
epoch_x = np.arange(1, num_epoch + 1)
for layer_idx, name in enumerate(weight_names):
    ax_epoch.plot(epoch_x,
                  [row[layer_idx] for row in persistent_zero_ratio_per_epoch],
                  label=name)
    ax_global.plot(epoch_x,
                   [row[layer_idx] for row in persistent_zero_ratio_global],
                   label=name)
ax_epoch.set(title='Persistent within each epoch', xlabel='Epoch',
             ylabel='Persistent near-zero ratio')
ax_global.set(title='Persistent since training started', xlabel='Epoch',
              ylabel='Persistent near-zero ratio')
for ax in (ax_epoch, ax_global):
    ax.legend()
    ax.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(save_dir / '3layer_persistent_zero_weights.png', dpi=150,
             bbox_inches='tight')

# log10(g1 / g2): negative values mean attenuation toward the input.
fig5, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
ratio_x = [0]
ratio_labels = [r'$g_1/g_2$']
for ax, ep in zip(axes, target_epochs):
    samples = gradient_ratio_history[ep]
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(samples)))
    for k, sample in enumerate(samples):
        ax.plot(ratio_x, np.log10(np.clip(sample, 1e-12, None)),
                marker='o', color=colors[k], alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f'Epoch {ep + 1}')
    ax.set_xticks(ratio_x)
    ax.set_xticklabels(ratio_labels)
    ax.set_xlabel('Backward transition')
    ax.set_ylabel(r'$\log_{10}(gradient\ ratio)$')
    ax.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig(save_dir / '3layer_gradient_change_rate.png', dpi=150,
             bbox_inches='tight')
plt.show()
