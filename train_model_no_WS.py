#!/usr/bin/env python3


import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from architectures.SimpleConvNet import SimpleConvNet
from dataset.CustomDataset import CustomDataset
from helpers.train_helpers import log_metric, get_device, get_dataset, get_optimizer, get_model, accuracy

import matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set(style="whitegrid", color_codes=True)
except ImportError:
    pass

def train(config):

    # Set the seed
    #torch.manual_seed(config['seed'])

    # We will run on CUDA if there is a GPU available
    device = get_device()

    # Configure the dataset, model and the optimizer based on the global `config` dictionary.
    training_loader, test_loader = get_dataset(config)
    model = get_model(device, config)
    optimizer = get_optimizer(model.parameters(), config)
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=config['logs_dir'])
    history = History()

    for epoch in range(1, config['num_epochs']+1):
        ### TRAIN ###

        # Enable training mode (automatic differentiation + batch norm)
        model.train()
        # collect train losses
        train_loss = Mean()
        # collect train accuracy
        train_accuracy = Mean()

        for batch_x, batch_y, batch_classes in training_loader:

            batch_x_1 = batch_x[:, 0, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device)
            batch_x_2 = batch_x[:, 1, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device)

            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)

            batch_y = batch_y.to(device)


            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction_1 = model(batch_x_1)
            prediction_2 = model(batch_x_2)
            loss = (criterion(prediction_1, batch_classes_1) + criterion(prediction_2, batch_classes_2)) / 2
            acc = accuracy(prediction_1.argmax(1) <= prediction_2.argmax(1), batch_y, argmax=False)
            loss.backward()

            # Do an optimizer step
            optimizer.step()

            # Store the statistics
            train_loss.update(loss.item(), n=len(batch_x))
            train_accuracy.update(acc.item(), n=len(batch_x))

        writer.add_scalar('Loss/train', train_loss.val(), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy.val(), epoch)

        ### EVALUATION ###
        # enable evaluation mode
        model.eval()
        # collect test losses
        test_loss = Mean()
        # collect test accuracy
        test_accuracy = Mean()

        for batch_x, batch_y, batch_classes in test_loader:

            batch_x_1 = batch_x[:, 0, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device)
            batch_x_2 = batch_x[:, 1, :, :].view(-1, 1, batch_x.size(2), batch_x.size(3)).to(device)

            batch_classes_1 = batch_classes[:, 0].to(device)
            batch_classes_2 = batch_classes[:, 1].to(device)

            batch_y = batch_y.to(device)

            prediction_1 = model(batch_x_1)
            prediction_2 = model(batch_x_2)
            loss = (criterion(prediction_1, batch_classes_1) + criterion(prediction_2, batch_classes_2)) / 2

            acc = accuracy(prediction_1.argmax(1) <= prediction_2.argmax(1), batch_y, argmax=False)

            test_loss.update(loss.item(), n=len(batch_x))
            test_accuracy.update(acc.item(), n=len(batch_x))

        
        writer.add_scalar('Loss/test', test_loss.val(), epoch)
        writer.add_scalar('Accuracy/test', test_accuracy.val(), epoch)

        # Log training stats
        history.update(epoch, train_loss.val(), test_loss.val(), train_accuracy.val(), test_accuracy.val())
        if config['verbose']:
            print(history)

    writer.close()

    return model, history



def accuracy(predicted_logits, reference, argmax=True):
    """Compute the ratio of correctly predicted labels"""
    if argmax:
        labels = torch.argmax(predicted_logits, 1)
    else:
        labels = predicted_logits

    correct_predictions = labels.float().eq(reference.float())
    return correct_predictions.sum().float() / correct_predictions.nelement()

class Mean:
    def __init__(self):
        self.avg = None
        self.counter = 0

    def update(self, val, n):
        self.counter += n
        if self.avg is None:
            self.avg = val
        else:
            delta = val - self.avg
            self.avg += delta * n / self.counter

    def val(self):
        return self.avg

class History:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.epochs = []
        self.losses_train = []
        self.losses_test = []
        self.accs_train = []
        self.accs_test = []

    def update(self, epoch, loss_train, loss_test, acc_train, acc_test):
        self.epochs += [epoch]
        self.losses_train += [loss_train]
        self.losses_test += [loss_test]
        self.accs_train += [acc_train]
        self.accs_test += [acc_test]

    def __repr__(self):
        return "epoch [{epoch:03d}] loss_train: {loss_train:.2e} loss_test: {loss_test:.2e} acc_train: {acc_train:.2e} acc_test: {acc_test:.2e}".format(epoch=self.epochs[-1], loss_train=self.losses_train[-1], loss_test=self.losses_test[-1], acc_train=self.accs_train[-1], acc_test=self.accs_test[-1])

    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(10,16))
        axs[0].plot(self.epochs, self.losses_test, marker="o", markersize=3, color="red", label="test loss");
        axs[0].plot(self.epochs, self.losses_train, marker="o", markersize=3, color="blue", label="train loss");
        axs[0].set_xlabel("epochs");axs[0].set_ylabel("Loss");axs[0].set_title("Loss", fontdict={"fontsize":20, "fontweight":1}, pad=15);
        axs[0].legend();

        axs[1].plot(self.epochs, self.accs_test, marker="o", markersize=3, color="red", label="test accuracy");
        axs[1].plot(self.epochs, self.accs_train, marker="o", markersize=3, color="blue", label="train accuracy");
        axs[1].set_xlabel("epochs");axs[1].set_ylabel("Accuracy");axs[1].set_title("Accuracy", fontdict={"fontsize":20, "fontweight":1}, pad=15);
        axs[1].legend();

        plt.show();
