from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

batch_size = 128
lr = 0.01
num_epochs = 100
weight_decay = 0.01
sample_every = 25
target_epochs = [0, 29, 44, 59, 70, 99]

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
data_root = Path('/home/yan/proj/data')
train_loader = DataLoader(
    FashionMNIST(data_root, train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
test_loader = DataLoader(
    FashionMNIST(data_root, train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, return_signals=False):
        z1 = self.fc1(self.flatten(x))
        h1 = F.relu(z1)
        logits = self.fc2(h1)
        return (logits, (z1, h1)) if return_signals else logits


def rms(tensor):
    """RMS computed with PyTorch's vector norm."""
    return (torch.linalg.vector_norm(tensor) / tensor.numel() ** 0.5).item()


torch.manual_seed(42)
model = MLP().to(device)
weight_params = [model.fc1.weight, model.fc2.weight]
optimizer = torch.optim.SGD([
    {'params': weight_params, 'weight_decay': weight_decay},
    {'params': [model.fc1.bias, model.fc2.bias], 'weight_decay': 0.0},
], lr=lr)

metric_names = (
    'forward_rms',
    'activation_grad_rms',
    'parameter_grad_rms',
    'relative_update',
    'relu_activity',
)
history = {name: [] for name in metric_names}
train_loss, test_loss, train_acc, test_acc = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    epoch_metrics = {name: [] for name in metric_names}
    loss_sum = correct = total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        sample = batch_idx % sample_every == 0

        optimizer.zero_grad()
        logits, (z1, h1) = model(inputs, return_signals=True)
        if sample:
            z1.retain_grad()
            h1.retain_grad()

        loss = F.cross_entropy(logits, targets)
        loss.backward()

        if sample:
            epoch_metrics['forward_rms'].append([rms(z1), rms(h1)])
            epoch_metrics['activation_grad_rms'].append([
                rms(z1.grad), rms(h1.grad)
            ])
            epoch_metrics['parameter_grad_rms'].append([
                rms(parameter.grad) for parameter in weight_params
            ])
            epoch_metrics['relu_activity'].append([
                torch.mean((z1 > 0).float()).item()
            ])
            weights_before = [parameter.detach().clone()
                              for parameter in weight_params]

        optimizer.step()

        if sample:
            epoch_metrics['relative_update'].append([
                (torch.linalg.vector_norm(parameter.detach() - before) /
                 torch.linalg.vector_norm(before).clamp_min(1e-12)).item()
                for parameter, before in zip(weight_params, weights_before)
            ])

        batch_items = targets.size(0)
        loss_sum += loss.item() * batch_items
        correct += torch.count_nonzero(logits.argmax(dim=1) == targets).item()
        total += batch_items

    for name in metric_names:
        history[name].append(epoch_metrics[name])
    train_loss.append(loss_sum / total)
    train_acc.append(correct / total)

    model.eval()
    loss_sum = correct = total = 0
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            batch_items = targets.size(0)
            loss_sum += F.cross_entropy(logits, targets).item() * batch_items
            correct += torch.count_nonzero(
                logits.argmax(dim=1) == targets
            ).item()
            total += batch_items

    test_loss.append(loss_sum / total)
    test_acc.append(correct / total)
    print(
        f'Epoch {epoch + 1:3d}: '
        f'train_loss={train_loss[-1]:.4f}, test_loss={test_loss[-1]:.4f}, '
        f'train_acc={train_acc[-1]:.3f}, test_acc={test_acc[-1]:.3f}'
    )


save_dir = Path(__file__).resolve().parent / 'figs'
save_dir.mkdir(exist_ok=True)


def plot_profiles(values, labels, ylabel, filename, use_log10=True):
    """Plot sampled batches for the selected epochs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(labels))
    for ax, epoch_idx in zip(axes.flat, target_epochs):
        samples = torch.as_tensor(values[epoch_idx])
        if use_log10:
            samples = samples.clamp_min(1e-12).log10()
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(samples)))
        for row, color in zip(samples, colors):
            ax.plot(x, row.numpy(), marker='o', color=color, alpha=0.7)
        ax.set(title=f'Epoch {epoch_idx + 1}', ylabel=ylabel)
        ax.set_xticks(x, labels)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / filename, dpi=150, bbox_inches='tight')


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
ax_loss.plot(train_loss, label='Train')
ax_loss.plot(test_loss, label='Test')
ax_loss.set(title='Loss', xlabel='Epoch', ylabel='Cross-entropy loss')
ax_acc.plot(train_acc, label='Train')
ax_acc.plot(test_acc, label='Test')
ax_acc.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
for ax in (ax_loss, ax_acc):
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(save_dir / '2layer_mlp.png', dpi=150, bbox_inches='tight')

plot_profiles(history['forward_rms'], ['z1', 'h1'],
              r'$\log_{10}(RMS(signal))$', '2layer_forward_signal_rms.png')
plot_profiles(history['activation_grad_rms'], ['grad z1', 'grad h1'],
              r'$\log_{10}(RMS(gradient))$',
              '2layer_activation_gradient_rms.png')
plot_profiles(history['parameter_grad_rms'], ['W1', 'W2'],
              r'$\log_{10}(RMS(\nabla W))$',
              '2layer_parameter_gradient_rms.png')
plot_profiles(history['relative_update'], ['W1', 'W2'],
              r'$\log_{10}(||\Delta W||/||W||)$',
              '2layer_relative_parameter_update.png')
plot_profiles(history['relu_activity'], ['ReLU 1'], r'$P(z_1 > 0)$',
              '2layer_relu_activity.png', use_log10=False)

plt.show()
