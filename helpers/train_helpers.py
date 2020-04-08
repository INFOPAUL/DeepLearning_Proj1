import torch
import torchvision

from architectures.Siamese import Siamese
from architectures.Siamese_no_WS import Siamese_no_WS
from architectures.SimpleConvNet import SimpleConvNet
from dataset.CustomDataset import CustomDataset


def accuracy(predicted_logits, reference, argmax=True):
    """Compute the ratio of correctly predicted labels"""
    if argmax:
        labels = torch.argmax(predicted_logits, 1)
    else:
        labels = predicted_logits
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(config):
    #train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    #data = generate_pair_sets(1000)

    #data_mean, data_stddev = data[0].mean().item()/ 255, data[0].std().item()/ 255

    #dataset statistics
    data_mean = 0.1307
    data_stddev = 0.3081


    train_transforms = [torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize([data_mean], [data_stddev])]
    if config['augmentation']:
        train_transforms = train_transforms + [torchvision.transforms.RandomCrop(28, padding=4),  torchvision.transforms.RandomHorizontalFlip()]
    transform_train = torchvision.transforms.Compose(train_transforms)

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([data_mean], [data_stddev]),
    ])

    training_set = CustomDataset('./data/mnist/', train=True, transform=transform_train, nb=1000)
    test_set = CustomDataset('./data/mnist/', train=False, transform=transform_test, nb=1000)


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
        'simple_conv':  lambda: SimpleConvNet(config['class_num'], config['channels_in']),
        'siamese': lambda: Siamese(config['class_num']),
        'siamese_no_WS': lambda: Siamese_no_WS(config['class_num'])
    }[config['model']]()

    model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model
