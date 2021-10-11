import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def init_data():
    data = np.genfromtxt(
        './data/email_spam.csv',
        delimiter=',',
        names=None,
    )
    samples = data.shape[0]
    data_array = data.view(np.float64).view().reshape(samples, -1)

    split = np.split(data_array, [-1], axis=1)
    features = split[0]

    labels = split[1]
    labels = labels.reshape((-1,)).astype('int64')

    scaled = [-1, -2, -3]
    for s in scaled:
        f = features[:, s]
        f /= (f.max() / 100)

    return features, labels


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(57, 2).double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    np_input, np_labels = init_data()
    input = torch.from_numpy(np_input)
    labels = torch.from_numpy(np_labels)
    print(input)
    print(f'{labels = }')

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20000):
        optimizer.zero_grad()

        outputs = net(input)
        # print(f'{outputs = }')
        loss = criterion(outputs, labels)
        loss.backward()
        print(f'{loss.item()=}')

        optimizer.step()

    outputs = net(input)
    _, predicted = torch.max(outputs, 1)
    print(f'{predicted = }')
    comparison = torch.eq(predicted, labels)
    print(f'{comparison.unique(return_counts=True) = }')
