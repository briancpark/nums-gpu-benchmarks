import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split


def create_model(in_features):
    mlp = nn.Sequential(
        nn.Linear(in_features, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 1),
        nn.Sigmoid()
    )

    return nn.DataParallel(mlp)

def load_dataset(n, flatten=False):
    # X, y = load_digits(return_X_y=True)
    X = np.random.rand(n, 10)
    y = np.random.choice([0, 1], size=(n, 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    # normalize x

    # X_train = X_train.astype(float)# / 255.
    # X_val = X_val.astype(float)# / 255.

    X_train, X_val, y_train, y_val = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)

    # we reserve the last 10000 training examples for validation
    # X_train, X_val = X_train[:-10000], X_train[-10000:]
    # y_train, y_val = y_train[:-10000], y_train[-10000:]


    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])

    return X_train, y_train, X_val, y_val

def train(n):
    X_train, y_train, X_val, y_val = load_dataset(n)

    model = create_model(X_train.shape[1])
    model.double()
    model.to(f'cuda:{model.device_ids[0]}')

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x = X_train.to(f'cuda:{model.device_ids[0]}')
        y = y_train.to(f'cuda:{model.device_ids[0]}').double()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        # print(f'[{epoch + 1}] loss: {running_loss:.3f}')
        running_loss = 0.0

        # with torch.no_grad():
        #     probs = model(X_val)
        #     predictions = (probs > 0).int()
            


def run_test(n):
    np.random.seed(267)
    train(n)    

if __name__ == "__main__":
    train(1000)
