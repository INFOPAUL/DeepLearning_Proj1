import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class CustomDataset(MNIST):
    
    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, download=True, transform=transform)

        input = self.data.view(-1, 1, 28, 28).float()
        target = self.targets

        input = torch.functional.F.avg_pool2d(input, kernel_size=2)
        a = torch.randperm(input.size(0))
        a = a[:2 * nb].view(nb, 2)
        self.pair_indices = a

        self.input = torch.cat((input[self.pair_indices[:, 0]], input[self.pair_indices[:, 1]]), 1)
        self.classes = target[self.pair_indices]
        self.target = (self.classes[:, 0] <= self.classes[:, 1]).long()

    def __len__(self):
        return len(self.input)
                      
    def __getitem__(self, index):
        # with last [0] we take img part of tuple (img, target)
        img1 = super().__getitem__(self.pair_indices[index][0])[0]  # get first img of the pair transformed
        img2 = super().__getitem__(self.pair_indices[index][1])[0]  # get second img of the pair transformed
        input = torch.cat((img1, img2), 0)
        input = torch.functional.F.avg_pool2d(input, kernel_size=2)

        target  = self.target[index]
        classes = self.classes[index]
        
        # return input, target, classes
        return input, target, classes

