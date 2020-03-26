#!/usr/bin/env python3


import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from architectures.SimpleConvNet import SimpleConvNet
from dataset.CustomDataset import CustomDataset
from helpers.mean import Mean
from helpers.train_helpers import log_metric, get_device, get_dataset, get_optimizer, get_model, accuracy


def train(config):

    # Set the seed
    #torch.manual_seed(config['seed'])

    # We will run on CUDA if there is a GPU available
    device = get_device()
    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset(config)

    model1 = get_model(device, config)
    model2 = get_model(device, config)
    optimizer = get_optimizer(list(model1.parameters()) + list(model2.parameters()),config)


    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='./logs')

    for epoch in range(config['num_epochs']):
        print('Epoch {:03d}'.format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model1.train()
        model2.train()
        # Update the optimizer's learning rate
        #scheduler.step(epoch)

        train_loss = Mean()
        train_accuracy = Mean()

        for batch_x, batch_y, batch_classes  in training_loader:
            batch_x, batch_y, batch_classes = batch_x.to(device), batch_y.to(device), batch_classes.to(device)


            # Compute gradients for the batch
            optimizer.zero_grad()

            prediction1 = model1(batch_x[:, 0, :, :].view(batch_x.size(0), 1, batch_x.size(2), batch_x.size(3)))
            prediction2 = model2(batch_x[:, 1, :, :].view(batch_x.size(0), 1, batch_x.size(2), batch_x.size(3)))

            loss1 = criterion(prediction1,  batch_classes[:,0])
            loss2 = criterion(prediction2,  batch_classes[:,1])
            loss = (loss1 + loss2) / 2

            acc = accuracy(prediction1.argmax(1) <= prediction2.argmax(1), batch_y, argmax=False)

            loss.backward()

            # Do an optimizer step
            optimizer.step()

            # Store the statistics
            train_loss.add(loss.item(), weight=len(batch_x))
            train_accuracy.add(acc.item(), weight=len(batch_x))
            # print(loss.item())
            # print(acc.item())

        # Log training stats
        log_metric(
            'accuracy',
            {'epoch': epoch, 'value': train_accuracy.val()},
            {'split': 'train'}
        )
        log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': train_loss.val()},
            {'split': 'train'}
        )

        # writer.add_scalar('Loss/train', mean_train_loss.value(), epoch)
        # writer.add_scalar('Accuracy/train', mean_train_accuracy.value(), epoch)

        # Evaluation
        model1.eval()
        model2.eval()

        test_loss = Mean()
        test_accuracy = Mean()
        for batch_x, batch_y, batch_classes in test_loader:
            batch_x, batch_y, batch_classes = batch_x.to(device), batch_y.to(device), batch_classes.to(device)

            prediction1 = model1(batch_x[:, 0, :, :].view(batch_x.size(0), 1, batch_x.size(2), batch_x.size(3)))
            prediction2 = model2(batch_x[:, 1, :, :].view(batch_x.size(0), 1, batch_x.size(2), batch_x.size(3)))

            loss1 = criterion(prediction1, batch_classes[:, 0])
            loss2 = criterion(prediction2, batch_classes[:, 1])
            loss = (loss1 + loss2) / 2

            acc = accuracy(prediction1.argmax(1) <= prediction2.argmax(1), batch_y, argmax=False)

            test_loss.add(loss.item(), weight=len(batch_x))
            test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            'accuracy',
            {'epoch': epoch, 'value': test_accuracy.val()},
            {'split': 'test'}
        )
        log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': test_loss.val()},
            {'split': 'test'}
        )

        # writer.add_scalar('Loss/test', mean_test_loss.value(), epoch)
        # writer.add_scalar('Accuracy/test', mean_test_accuracy.value(), epoch)
        writer.flush()

    writer.close()