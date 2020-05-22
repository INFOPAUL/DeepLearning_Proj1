import torch
from torch import nn
from torch.nn import functional as F
from architectures.SimpleConvNet import SimpleConvNet
from train import Mean
from dataset.CustomDataset import CustomDataset


class WeightSharingAuxLosses(nn.Module):
    """
    This model:
    - implements weight sharing between two images (WeightSharing)
    - uses auxiliary losses for the performance measure (AuxLosses)

    Parameters
    ----------
    class_num: int
        Number of classes
    channels_in: int
        Number of image channels
    
    Model architecture:
        - block1: uses previously implemented SimpleConvNet to classify the digit on the first image,
        it will be used for both of images
        - fc1: fully-connected layer that uses two-channeled image and outputs the feature vector
        that is used to classify whether the first digit is less than or equal to the second digit.
    
    Auxiliary loss:
        - cross entropy loss of predicting the first digit +
        - cross entropy loss of predicting the second digit +
        - cross entropy loss of predicting their relation (<=)
    """
    def __init__(self, class_num=10, channels_in=1):
        super().__init__()
        self.block1 = SimpleConvNet(class_num=class_num, channels_in=channels_in)

        self.fc1 = nn.Linear(20, class_num)

    def forward(self, x):
        out1, _, _ = self.block1(x)
        # size: 1000, 10
        _, out2, _ = self.block1(x)
        # size: 1000, 10

        cat = torch.cat([out1, out2], dim=1)
        # size: 1000, 20
        out = self.fc1(cat)
        # size: 1000, 10

        return out1, out2, out

    def train_(self, training_loader, device, optimizer):
        """Train the model
        
        Parameters
        ----------
        training_loader: generator
            Training data generator
        device: str
            cuda or cpu
        optimizer: callable
            One of the PyTorch optimizers

        Returns
        -------
        train_loss: Mean
            Class object that collects the train loss
        train_accuracy: Mean
            Class object that collects the train accuracy
        """
        # Train loss for this epoch
        train_loss = Mean()
        # Train accuracy for this epoch
        train_accuracy = Mean()
        
        for batch_x, batch_y, batch_classes in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_classes1 = batch_classes[:, 0].to(device)
            batch_classes2 = batch_classes[:, 1].to(device)

            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            predict_digit1, predict_digit2, predict_y = self(batch_x)
            loss = F.cross_entropy(predict_y, batch_y) + \
                   F.cross_entropy(predict_digit1, batch_classes1) + \
                   F.cross_entropy(predict_digit2, batch_classes2)
            acc = self.accuracy_(predict_y, batch_y)
            
            # Backward propagation of gradients
            loss.backward()

            # Do an optimizer step (updating parameters)
            optimizer.step()

            # Store the statistics
            train_loss.update(loss.item(), n=len(batch_x))
            train_accuracy.update(acc.item(), n=len(batch_x))

        return train_loss, train_accuracy

    def eval_(self, test_loader, device):
        """Evaluate the model
        
        Parameters
        ----------
        test_loader: generator
            Test data generator
        device: str
            cuda or cpu

        Returns
        -------
        test_loss: Mean
            Class object that collects the test loss
        test_accuracy: Mean
            Class object that collects the test accuracy
        """
        # Test loss for this epoch
        test_loss = Mean()
        # Test accuracy for this epoch
        test_accuracy = Mean()

        for batch_x, batch_y, batch_classes in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_classes1 = batch_classes[:, 0].to(device)
            batch_classes2 = batch_classes[:, 1].to(device)
            
            # Calculate loss and accuracy
            predict_digit1, predict_digit2, predict_y = self(batch_x)
            loss = F.cross_entropy(predict_y, batch_y) + \
                   F.cross_entropy(predict_digit1, batch_classes1) + \
                   F.cross_entropy(predict_digit2, batch_classes2)
            acc = self.accuracy_(predict_y, batch_y)

            # Store the statistics
            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        return test_loss, test_accuracy

    def accuracy_(self, predicted_logits, reference, argmax=True):
        """Compute the ratio of correctly predicted labels
        
        Parameters
        ----------
        predicted_logits: tensor
            One hot label of the predicted value
        reference: tensor
            Target value
        """
        labels = torch.argmax(predicted_logits, 1)
        correct_predictions = labels.eq(reference)
        return correct_predictions.sum().float() / correct_predictions.nelement()


class WeightSharingAuxLossesDataset(CustomDataset):
    """Dataset generator for the WeightSharingAuxLosses model"""

    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, transform=transform, nb=nb)

    def __getitem__(self, index):
        return super().__getitem__(index)