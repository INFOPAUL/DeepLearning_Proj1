#!/usr/bin/env python3
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from architectures.SimpleConvNet import SimpleConvNet
from dataset.CustomDataset import CustomDataset
import matplotlib.pyplot as plt
try:import seaborn as sns; sns.set(style="whitegrid", color_codes=True)
except ImportError: pass


def train(config):
    # Set the seed
    #torch.manual_seed(config['seed'])

    # We will run on CUDA if there is a GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure the dataset, model, optimizer and criterion based on the global `config` dictionary.
    model = config['model']
    model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
    criterion = config['criterion']

    training_loader, test_loader = get_dataset(config)

    print("Number of model parameters: {params}".format(params=sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=config['logs_dir'])
    # Initialize our history writer
    history = History()

    for epoch in range(1, config['num_epochs']+1):
        ### TRAIN ###
        # Enable training mode (automatic differentiation + batch norm)
        model.train()
        train_loss, train_accuracy = model.train_(training_loader, device, optimizer, criterion)

        writer.add_scalar('Loss/train', train_loss.val(), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy.val(), epoch)

        ### EVALUATION ###
        # Enable evaluation mode
        model.eval()
        test_loss, test_accuracy = model.eval_(test_loader, device, criterion)
        
        writer.add_scalar('Loss/test', test_loss.val(), epoch)
        writer.add_scalar('Accuracy/test', test_accuracy.val(), epoch)

        # Log training stats
        history.update(epoch, train_loss.val(), test_loss.val(), train_accuracy.val(), test_accuracy.val())
        if config['verbose']:
            print(history)

    writer.close()

    return model, history


def get_dataset(config):
    # dataset statistics: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    data_mean = 0.1307
    data_stddev = 0.3081

    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize([data_mean], [data_stddev])]
    
    # TRAIN transforms
    train_transforms = transforms + config['augmentation']*[torchvision.transforms.RandomCrop(28, padding=4),  
                                                            torchvision.transforms.RandomHorizontalFlip()]
    transform_train = torchvision.transforms.Compose(train_transforms)

    # TEST transforms
    transform_test = torchvision.transforms.Compose(transforms)


    training_set = config['dataset']('./data/mnist/', train=True, transform=transform_train, nb=1000)
    test_set     = config['dataset']('./data/mnist/', train=False, transform=transform_test, nb=1000)

    if config['verbose']:
        print(training_set)
        print(test_set)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    return training_loader, test_loader


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
