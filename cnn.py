import gzip
import pickle
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


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

data = (x_train, y_train, x_valid, y_valid)

x_train, y_train, x_valid, y_valid = map(torch.tensor, data)

batch_size = 64
learning_rate = 0.005
epochs = 10

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)

# The batch is double 'cause there is not backpropagation
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)


def get_model():
    model = Mnist_CNN()
    loss_func = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model, opt, loss_func


model, opt, loss_func = get_model()


def fit():
    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(train_dl):
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(f'epoch : {epoch}, valid_loss : {valid_loss / len(valid_dl)}')


fit()
