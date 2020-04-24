import gzip
import pickle
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super(Mnist_Logistic, self).__init__()
        # self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        # self.bias = nn.Parameter(torch.zeros(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, x):
        # return x @ self.weights + self.bias
        return self.lin(x)


data_path = Path('data')
path = data_path / 'mnist'

path.mkdir(parents=True, exist_ok=True)

url = "http://deeplearning.net/data/mnist/"
filename = "mnist.pkl.gz"

file_path = path / filename

if not file_path.exists():
    content = requests.get(url + filename).content
    file_path.open('wb').write(content)

with gzip.open(file_path.as_posix(), 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    ((x_train, y_train), (x_valid, y_valid), _) = data

# img = pyplot.imshow(x_train[0].reshape(28, 28), cmap='gray')
# pyplot.show()

data = (x_train, y_train, x_valid, y_valid)

x_train, y_train, x_valid, y_valid = map(torch.tensor, data)

n_samples, dim_sample = x_train.shape

loss_func = F.cross_entropy

batch_size = 4
learning_rate = 0.1
epochs = 4

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)

# The batch is double 'cause there is not backpropagation
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)


def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=learning_rate)


model, opt = get_model()


def fit():
    for epoch in tqdm(range(epochs)):
        # for i in range((n_samples - 1) // batch_size + 1):
        #     start_i = i * batch_size
        #     end_i = start_i + batch_size
        #     xb = x_train[start_i:end_i]
        #     yb = y_train[start_i:end_i]
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            # with torch.no_grad():
            #     for p in model.parameters():
            #         p -= p.grad * learning_rate
            #         model.zero_grad()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(f'epoch : {epoch}, valid_loss : {valid_loss / len(valid_dl)}')


fit()
