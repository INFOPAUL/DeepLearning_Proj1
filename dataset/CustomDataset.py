import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class CustomDataset(MNIST):
    def __init__(self, root, train=True, transform=None, nb=1000):
        super(CustomDataset, self).__init__(root, train = train, download = True, transform = transform)

        input = self.data
        target = self.targets

        a = torch.randperm(input.size(0))
        self.input_indexes = a[:2 * nb].view(nb, 2)
        self.classes = target[self.input_indexes]
        self.target = (self.classes[:, 0] <= self.classes[:, 1]).long()

    def __len__(self):
        return len(self.input_indexes)

    def __getitem__(self, index):
        target = int(self.target[index])
        classes = self.classes[index]
        img1 = super(CustomDataset, self).__getitem__(self.input_indexes[index][0])[0]
        img2 = super(CustomDataset, self).__getitem__(self.input_indexes[index][1])[0]

        return torch.cat((img1, img2), 0), target, classes