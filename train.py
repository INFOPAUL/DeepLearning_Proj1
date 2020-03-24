#!/usr/bin/env python3


import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from architectures.SimpleConvNet import SimpleConvNet
from dataset.CustomDataset import CustomDataset
from helpers.mean import Mean


def train(config):

    # Set the seed
    #torch.manual_seed(config['seed'])

    # We will run on CUDA if there is a GPU available
    device = get_device()
    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset(config)
    model = get_model(device, config)
    optimizer = get_optimizer(model.parameters(),config)
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='./logs')

    for epoch in range(config['num_epochs']):
        print('Epoch {:03d}'.format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Update the optimizer's learning rate
        #scheduler.step(epoch)

        train_loss = Mean()
        train_accuracy = Mean()

        for batch_x, batch_y in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)


            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
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
        model.eval()
        test_loss = Mean()
        test_accuracy = Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
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


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))

def get_device():
    return torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def get_dataset(config):
    #train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    #data = generate_pair_sets(1000)

    #data_mean, data_stddev = data[0].mean().item()/ 255, data[0].std().item()/ 255

    #dataset statistics
    data_mean = 0.1307
    data_stddev = 0.3081

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([data_mean], [data_stddev]),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([data_mean], [data_stddev]),
    ])

    training_set = CustomDataset('./data/mnist/', train=True, transform=transform_train)
    test_set = CustomDataset('./data/mnist/', train=False, transform=transform_test)


    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config['batch_size'],
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
    )

    return training_loader, test_loader


def get_optimizer(model_parameters, config):
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config['optimizer_learning_rate']
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=config['optimizer_learning_rate']
        )
    else:
        raise ValueError('Unexpected value for optimizer')

    return optimizer


def get_model(device, config):
    model = {
        'simple_conv':  lambda: SimpleConvNet(),
        #'simple_conv':  lambda: SimpleConvNet(),
    }[config['model']]()

    model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model