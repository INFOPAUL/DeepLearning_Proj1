import torch
from torch import nn
from torch.nn import functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, class_num, channels_in):
        super(SimpleConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, 1000)

        self.fc2 = nn.Linear(1000, class_num)

    def forward(self, x):
        out = self.block1(x)

        out = self.block2(out)

        # Reshape (batch, 1024)
        out = out.reshape(out.size(0), -1)

        # Relu activation of last layer
        out = F.relu(self.fc1(out.view(-1, 7 * 7 * 64)))

        out = self.fc2(out)
        return out