import torch
from torch import nn
from torch.nn import functional as F

from architectures.SimpleConvNet import SimpleConvNet


class Siamese_no_WS(nn.Module):
    def __init__(self, class_num):
        super(Siamese_no_WS, self).__init__()
        self.block1 = SimpleConvNet(class_num = 10, channels_in = 1)
        self.block2 = SimpleConvNet(class_num = 10, channels_in = 1)

        self.fc1 = nn.Linear(20, class_num)

    def forward(self, x):
        x1 = x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3))
        x2 = x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3))

        out1 = self.block1(x1)
        out2 = self.block2(x2)

        cat = torch.cat([out1,out2], dim=1)
        out = self.fc1(cat)

        return out