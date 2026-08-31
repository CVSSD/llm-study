import time
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt

resize = (32, 32)
trans = transforms.Compose([transforms.Resize(resize),
                            transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(
    root='/home/yan/proj/data',
    train=True,
    download=True,
    transform=trans
)
val = torchvision.datasets.FashionMNIST(
    root='/home/yan/proj/data',
    train=False,
    download=True,
    transform=trans
)

train_data = train.data.reshape(60000,-1).float()/255.0
val_data = val.data.reshape(10000,-1).float()/255.0

train_target = train.targets
val_target = val.targets
del train
del val

#Build loss function
def loss_log_softmax_likelihood(o_hat, y, accuracy=False):
     o_max = o_hat.max(dim=1, keepdim=True).values
     y_hat = o_hat - torch.log(torch.sum(torch.exp(o_hat - o_max), dim=1, keepdim=True)) - o_max
     labels = y.argmax(dim=1)
     if accuracy:
          prediction = y_hat.argmax(dim=1)
          labels = y.argmax(dim=1)

          return (prediction == labels).float().mean()
     return -y_hat[torch.arange(len(o_hat)), labels].mean()


# SGD Stohastic Gradient Decent
def miniBatch(batchSize, numOfTrainSample):
    index = list(range(0,numOfTrainSample))
    random.shuffle(index)
    batches = []

    for i in range(0, numOfTrainSample, batchSize):
        batch = index[i:i+batchSize]
        batches.append(batch)
    if numOfTrainSample % batchSize != 0:
        batches.pop()

    return batches

def targets_vectorize(targets):
     targets_one_hot = torch.zeros(len(targets), torch.max(targets) + 1)
     for i in range(len(targets)):
          targets_one_hot[i][targets[i]] = 1
     return targets_one_hot


w = torch.randn(train_data.size()[1], 10, dtype=torch.float32, requires_grad=True)
b = torch.zeros(10, dtype=torch.float32, requires_grad=True)
lr = 0.01
batchSize = 256
numTrainSample = train_data.size()[0]
train_target = targets_vectorize(train_target)
val_target = targets_vectorize(val_target)
train_data = train_data
val_data = val_data

batches = miniBatch(batchSize, train_data.size()[0])

numOfEpoch = 5000
train_loss = []
train_loss_track = []
val_loss_track = []
val_accuracy = []
#print(w)
#print(b)
for epochIndex in range(0,numOfEpoch):
    batches = miniBatch(batchSize, numTrainSample)
    for batchIndex in batches:
        X_batch = train_data[batchIndex].to(torch.float32)
        y_batch = train_target[batchIndex].to(torch.float32)

        o_hat = X_batch @ w + b
        loss = loss_log_softmax_likelihood(o_hat, y_batch)
        train_loss.append(loss.item())
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()
    train_loss_track.append(sum(train_loss) / len(train_loss))
    train_loss = []

    with torch.no_grad():
        x_val = val_data.to(torch.float32)

        o_hat = x_val @ w + b
        loss = loss_log_softmax_likelihood(o_hat, val_target) 
        val_loss_track.append(loss)
        val_accuracy.append(loss_log_softmax_likelihood(o_hat, val_target, accuracy=True))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.plot(np.log10(train_loss_track), label="Training Loss")
plt.plot(np.log10(val_loss_track), label="Validation Loss")
plt.plot(val_accuracy, label="Validation Accuracy")
#plt.plot(train_loss_track, label="Training Loss")
#plt.plot(val_loss_track, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
#plt.yscale('log')

plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.show()

print(train_loss_track[-1])
print(val_loss_track[-1])
print(val_accuracy[-1])