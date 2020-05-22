import torch
from torch import nn
from torch.nn import functional as F
from architectures.SimpleConvNet import SimpleConvNet
from train import Mean
from dataset.CustomDataset import CustomDataset


class NoWeightSharing(nn.Module):
    """
    This model:
    - does not implement weight sharing between two images (NoWeightSharing)
    - does not use auxiliary losses for the performance measure

     Parameters
    ----------
    class_num: int
        Number of classes
    channels_in: int
        Number of image channels
    
    Model architecture:
        - block1: uses previously implemented SimpleConvNet to classify the digit on the first image
        - block2: uses previously implemented SimpleConvNet to classify the digit on the second image
        these two block don't share the weights
        - fc1: fully-connected layer that uses two-channeled image and outputs the feature vector
        that is used to classify whether the first digit is less than or equal to the second digit.
    
    Loss:
        - Cross entropy calculated between target and predicted feature vector
    """
    def __init__(self, class_num=10, channels_in=1):
        super().__init__()
        self.block1 = SimpleConvNet(class_num=class_num, channels_in=channels_in)
        self.block2 = SimpleConvNet(class_num=class_num, channels_in=channels_in)

        self.fc1 = nn.Linear(20, class_num)

    def forward(self, x):
        out1, _, _ = self.block1(x)
        _, out2, _ = self.block2(x)

        cat = torch.cat([out1,out2], dim=1)
        out = self.fc1(cat)

        return out

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

            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            prediction = self(batch_x)
            # prediction size: 1000, 10; 
            # batch_y size:    1000
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
            
            prediction = self(batch_x)
            loss = F.cross_entropy(prediction, batch_y)

            acc = self.accuracy_(prediction, batch_y)

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

class NoWeightSharingDataset(CustomDataset):
    """Dataset generator for the NoWeightSharing model"""
    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, transform=transform, nb=nb)

    def __getitem__(self, index):
        return super().__getitem__(index)