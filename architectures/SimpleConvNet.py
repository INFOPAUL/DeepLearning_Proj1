import torch
from torch import nn
from torch.nn import functional as F
from train import Mean
from dataset.CustomDataset import CustomDataset

class SimpleConvNet(nn.Module):
    """
    This model represents a submodel that will be used in 4 models where we will be testing the 
    effects of the weight sharing as well as the use of auxilliary losses.

    Parameters
    ----------
    class_num: int
        Number of classes
    channels_in: int
        Number of image channels
    
    Model architecture:
        - feature_extract: set of layers that extracts the imformation from digit images, it implements
        Conv2d->ReLU->MaxPool2d->Conv2d->ReLU->MaxPool2d layers
        - digit classifier: used to classify the two feature vectors of two corresponding images and outputs
        a 10-dim vector in the end. It implements: Linear->ReLU->Linear layers

    Loss is calculated using the mean of cross entropy loss between GT first image and predicted one + GT second image and predicted one
    """
    def __init__(self, class_num=10, channels_in=1):
        super().__init__()
        self.features_extract = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.digit_classifier = nn.Sequential(
            nn.Linear(16 * 64, 1000),
            nn.ReLU(),
            nn.Linear(1000, class_num)
        )

    def forward(self, x):
        # size: 1000, 2, 14, 14
        x1 = x[:, 0, :, :].view(-1, 1, x.size(2), x.size(3))
        x2 = x[:, 1, :, :].view(-1, 1, x.size(2), x.size(3))

        # size: 1000, 1, 14, 14
        out1 = self.features_extract(x1).view(x.size(0), -1)  
        # size: 1000, 1024
        digit1 = self.digit_classifier(out1)
        # size: 1000, 10

        # size: 1000, 1, 14, 14
        out2 = self.features_extract(x2).view(x.size(0), -1)  
        # size: 1000, 1024
        digit2 = self.digit_classifier(out2)
        # size: 1000, 10

        return digit1, digit2, digit1.argmax(1) <= digit2.argmax(1)

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
            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)

            # Set gradients to zero and Compute gradients for the batch
            optimizer.zero_grad()

            # Calculate loss and accuracy
            predict_class_1, predict_class_2, predict_y = self(batch_x)
            loss = (F.cross_entropy(predict_class_1, batch_classes_1) + F.cross_entropy(predict_class_2, batch_classes_2)) / 2
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
            batch_x = batch_x.to(device)
            
            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)
            batch_y = batch_y.to(device)

            # Calculate loss and accuracy
            predict_class_1, predict_class_2, predict_y = self(batch_x)
            loss = (F.cross_entropy(predict_class_1, batch_classes_1) + F.cross_entropy(predict_class_2, batch_classes_2)) / 2
            acc = self.accuracy_(predict_y, batch_y)
           
            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        return test_loss, test_accuracy

    def accuracy_(self, predicted_logits, reference):
        """Compute the ratio of correctly predicted labels
        
        Parameters
        ----------
        predicted_logits: tensor
            One hot label of the predicted value
        reference: tensor
            Target value
        """
        correct_predictions = predicted_logits.float().eq(reference.float())
        return correct_predictions.sum().float() / correct_predictions.nelement()


class SimpleConvNetDataset(CustomDataset):
    """Dataset generator for the SimpleConvNet model"""

    def __init__(self, root, train=True, transform=None, nb=1000):
        super().__init__(root, train=train, transform=transform, nb=nb)

    def __getitem__(self, index):
        return super().__getitem__(index)
