import torch
from torch import nn
from torch.nn import functional as F
from train import Mean
from dataset.CustomDataset import CustomDataset


class Linear(nn.Module):
    
    def __init__(self, channels_in=2*14*14, channels_out=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(channels_in, channels_out)
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return self.block1(out)

    def train_(self, training_loader, device, optimizer):
        # Train loss for this epoch
        train_loss = Mean()
        # Train accuracy for this epoch
        train_accuracy = Mean()
        
        for batch_x, batch_y, batch_classes in training_loader:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).float()

            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            prediction = self(batch_x).float()
            loss = F.cross_entropy(prediction, batch_y)
            acc = self.accuracy_(prediction, batch_y)
            
            # Backward propagation of gradients
            loss.backward()

            # Do an optimizer step (updating parameters)
            optimizer.step()

            # Store the statistics
            train_loss.update(loss.item(), n=len(batch_x))
            train_accuracy.update(acc.item(), n=len(batch_x))

        return train_loss, train_accuracy

    def eval_(self, test_loader, device):
        # Test loss for this epoch
        test_loss = Mean()
        # Test accuracy for this epoch
        test_accuracy = Mean()

        for batch_x, batch_y, batch_classes in test_loader:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).float()
            
            prediction = self(batch_x).float()
            loss = F.cross_entropy(prediction, batch_y)

            acc = self.accuracy_(prediction, batch_y)

            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        return test_loss, test_accuracy

    def accuracy_(self, y_pred, y_target):
        y_pred = y_pred.clone()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        correct_predictions = y_pred.eq(y_target)
        return correct_predictions.sum().float() / correct_predictions.nelement()



class LinearDataset(CustomDataset):

    def __init__(self, root, train=True, transform=None, nb=1000):
        """
        input: N x 392
        target: N x 1
        """
        super().__init__(root, train=train, transform=transform, nb=nb)
        self.input = self.input.reshape(self.input.size(0), -1)#.unsqueeze(1)
        self.target = self.target.unsqueeze(1)

    def __getitem__(self, index):
        return super().__getitem__(index)
        
