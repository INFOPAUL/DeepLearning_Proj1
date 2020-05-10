import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class CustomDataset(MNIST):
    
    def __init__(self, root, train=True, transform=None, nb=1000):
        """
        MNIST dataset returns images shape: Nx28x28

        data: images
            N x 14 x 14
        input: pairs of images to compare
            N x 2 x 14 x 14
        target: is the first image number greater than the number on the second image - 0, otherwise - 1
            N
        classes: the class of each image
            N x 2
        """
        super().__init__(root, train=train, download=True, transform=transform)

        self.data = self.data.view(-1, 1, 28, 28).float()
        self.data = torch.functional.F.avg_pool2d(self.data, kernel_size=2)
        a = torch.randperm(self.data.size(0))
        a = a[:2 * nb].view(nb, 2)
        self.pair_indices = a

        self.input = torch.cat((self.data[self.pair_indices[:, 0]], self.data[self.pair_indices[:, 1]]), 1)
        self.classes = self.targets[self.pair_indices]
        self.target = (self.classes[:, 0] <= self.classes[:, 1]).long()

        self.data = self.data.view(-1, 14, 14).float()

    def __len__(self):
        return len(self.input)
                      
    def __getitem__(self, index):
        input = self.input[index]
        target  = self.target[index]
        classes = self.classes[index]
        
        # return input, target, classes
        return input, target, classes


    def __repr__(self):
        ret_val = ""
        ret_val += "Shapes for {} set:\n".format('TRAIN' if self.train else 'TEST')
        ret_val += "\tall data => {}\n".format(self.data.shape)
        ret_val += "\tinput    => {}\n".format(self.input.shape)
        ret_val += "\ttarget   => {}\n".format(self.target.shape)
        ret_val += "\tclasses  => {}\n".format(self.classes.shape)
        return ret_val

