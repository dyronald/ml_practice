import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print('.')
        print('.')
        print('.')
        x = self.conv1(x)
        print(f'1: {x.shape}')
        # print(x)

        x = F.relu(x)
        print(f'2: {x.shape}')

        x = F.max_pool2d(x, (2,2))
        print(f'3: {x.shape}')

        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.conv2(x)
        print(f'4: {x.shape}')

        x = F.relu(x)
        print(f'5: {x.shape}')

        x = F.max_pool2d(x, 2)
        print(f'6: {x.shape}')

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        print(f'7: {x.shape}')

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)

params = list(net.parameters())
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)

net.zero_grad()
out.backward(torch.randn(1, 10))

optimizer.step()
