import torch
import random
from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val              
        self.X = d2l.randn(n, len(w))
        noise = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)

    def get_dataloader(self, train):
        if train:
            ind = list(range(0, self.num_train))
            # The example run in random order
            random.shuffle(ind)
        else:
            ind =  list(range(self.num_train, self.num_train+self.num_val))
        for i in range(0, len(ind), self.batch_size):
            batch_ind = d2l.tensor(ind[i: i+self.batch_size])
            yield self.X[batch_ind], self.y[batch_ind]
