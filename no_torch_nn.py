import gzip
import math
import pickle
from pathlib import Path

import numpy as numpy
import requests
import torch
from matplotlib import pyplot
from tqdm import tqdm

# mnist is a dataset of black-white img of digits

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

# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# pytorch model

weights_dim = 10

# initalize with the Xavier initialization: rand / sqrt(n)
weights = torch.randn(dim_sample, weights_dim) / math.sqrt(n_samples)
weights.requires_grad_()  # other form to require_grad
bias = torch.zeros(weights_dim, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(x):
    return log_softmax(x @ weights + bias)


batch_size = 3
x = x_train[:batch_size]
preds = log_softmax(x @ weights + bias)


# in the reality the log part is in log_softmax
def negative_log_likehood(input, target):
    # Only select the elements correspondig to the correct ones
    return -input[range(target.shape[0]), target].mean()


loss_func = negative_log_likehood
y = y_train[:batch_size]
loss = loss_func(preds, y)
print(f'The loss function is : {loss} ')


def accuracy(outputs, y):
    preds = torch.argmax(outputs, dim=1)
    return (preds == y).float().mean()


print(accuracy(preds, y))

learning_rate = 0.1
epochs = 4

for epoch in tqdm(range(epochs)):
    for i in range((n_samples - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        # at every operation the gradient is recorded,
        # in this case must be avoid with no_grad
        with torch.no_grad():
            weights -= weights.grad * learning_rate
            bias -= bias.grad * learning_rate

            # the gradient accumulate by default, need reset
            weights.grad.zero_()
            bias.grad.zero_()

outputs = model(x_valid)
loss = loss_func(outputs, y_valid)
acc = accuracy(outputs, y_valid)
print(f'The loss is : {loss}, accuracy : {acc}')
