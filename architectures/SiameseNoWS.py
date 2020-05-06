import torch
from torch import nn
from torch.nn import functional as F
from architectures.SimpleConvNet import SimpleConvNet
from train import Mean
from dataset.CustomDataset import CustomDataset


class SiameseNoWS(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()
        self.block1 = SimpleConvNet(class_num=10, channels_in=1)
        self.block2 = SimpleConvNet(class_num=10, channels_in=1)

        self.fc1 = nn.Linear(20, class_num)

    def forward(self, x):
        x1 = x[:, 0, :, :].view(x.size(0), 1, x.size(2), x.size(3))
        x2 = x[:, 1, :, :].view(x.size(0), 1, x.size(2), x.size(3))

        out1 = self.block1(x1)
        out2 = self.block2(x2)

        cat = torch.cat([out1,out2], dim=1)
        out = self.fc1(cat)

        return out

    def train_(self, training_loader, device, optimizer, criterion):
        # Train loss for this epoch
        train_loss = Mean()
        # Train accuracy for this epoch
        train_accuracy = Mean()
        
        for batch_x, batch_y, batch_classes in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)


            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            prediction = self(batch_x)
            loss = criterion(prediction, batch_y)
            acc = self.accuracy_(prediction, batch_y)
            
            # Backward propagation of gradients
            loss.backward()

            # Do an optimizer step (updating parameters)
            optimizer.step()

            # Store the statistics
            train_loss.update(loss.item(), n=len(batch_x))
            train_accuracy.update(acc.item(), n=len(batch_x))

        return train_loss, train_accuracy

    def eval_(self, test_loader, device, criterion):
        # Test loss for this epoch
        test_loss = Mean()
        # Test accuracy for this epoch
        test_accuracy = Mean()

        for batch_x, batch_y, batch_classes in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            prediction = self(batch_x)
            loss = criterion(prediction, batch_y)

            acc = self.accuracy_(prediction, batch_y)

            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        return test_loss, test_accuracy

    def accuracy_(self, predicted_logits, reference, argmax=True):
        """Compute the ratio of correctly predicted labels"""
        if argmax:
            labels = torch.argmax(predicted_logits, 1)
        else:
            labels = predicted_logits

        correct_predictions = labels.eq(reference)
        return correct_predictions.sum().float() / correct_predictions.nelement()

class SiameseNoWSDataset(CustomDataset):

    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, transform=transform, nb=nb)

    def __getitem__(self, index):
        return super().__getitem__(index)