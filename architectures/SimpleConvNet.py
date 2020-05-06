import torch
from torch import nn
from torch.nn import functional as F
from train import Mean
from dataset.CustomDataset import CustomDataset

class SimpleConvNet(nn.Module):

    def __init__(self, class_num, channels_in):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(16 * 64, 1000)  # 16

        self.fc2 = nn.Linear(1000, class_num)

    def forward(self, x):
        out = self.block1(x)

        out = self.block2(out)

        # Reshape (batch, 1024)
        out = out.reshape(out.size(0), -1)

        # Relu activation of last layer
        out = F.relu(self.fc1(out.view(-1, 16 * 64)))  # 16

        out = self.fc2(out)
        return out

    def train_(self, training_loader, device, optimizer, criterion):
        # Train loss for this epoch
        train_loss = Mean()
        # Train accuracy for this epoch
        train_accuracy = Mean()
        
        for batch_x, batch_y, batch_classes in training_loader:

            batch_x_1 = batch_x[:, 0, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device).float()
            batch_x_2 = batch_x[:, 1, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device).float()

            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)
            batch_y = batch_y.to(device)

            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            prediction_1 = self(batch_x_1)
            prediction_2 = self(batch_x_2)
            loss = (criterion(prediction_1, batch_classes_1) + criterion(prediction_2, batch_classes_2)) / 2
            acc = self.accuracy_(prediction_1.argmax(1) <= prediction_2.argmax(1), batch_y, argmax=False)
            
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

            batch_x_1 = batch_x[:, 0, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device).float()
            batch_x_2 = batch_x[:, 1, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device).float()

            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)

            batch_y = batch_y.to(device)

            prediction_1 = self(batch_x_1)
            prediction_2 = self(batch_x_2)
            loss = (criterion(prediction_1, batch_classes_1) + criterion(prediction_2, batch_classes_2)) / 2

            acc = self.accuracy_(prediction_1.argmax(1) <= prediction_2.argmax(1), batch_y, argmax=False)

            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        return test_loss, test_accuracy

    def accuracy_(self, predicted_logits, reference, argmax=True):
        """Compute the ratio of correctly predicted labels"""
        if argmax:
            labels = torch.argmax(predicted_logits, 1)
        else:
            labels = predicted_logits

        correct_predictions = labels.float().eq(reference.float())
        return correct_predictions.sum().float() / correct_predictions.nelement()


class SimpleConvNetDataset(CustomDataset):

    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, transform=transform, nb=nb)

    def __getitem__(self, index):
        return super().__getitem__(index)
