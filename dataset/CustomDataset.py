import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_tensor, train=True, transform=None):
        if train:
            self.data = data_tensor[0]
            self.targets = data_tensor[1]
        else:
            self.data = data_tensor[3]
            self.targets = data_tensor[4]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img[0].numpy(), mode='L')
        img2 = Image.fromarray(img[1].numpy(), mode='L')

        if self.transform is not None:
            img[0] = self.transform(img1)
            img[1] = self.transform(img2)

        return img, target