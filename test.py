import torch
import train
from datetime import datetime
from architectures.SimpleConvNet import SimpleConvNet, SimpleConvNetDataset
from architectures.WeightSharing import WeightSharing, WeightSharingDataset
from architectures.WeightSharing_AuxLosses import WeightSharingAuxLosses, WeightSharingAuxLossesDataset
from architectures.NoWeightSharing import NoWeightSharing, NoWeightSharingDataset
from architectures.NoWeightSharing_AuxLosses import NoWeightSharingAuxLosses, NoWeightSharingAuxLossesDataset


"""
NETWORK choices:
    1 - SimpleConvNet
    2 - WeightSharing
    3 - WeightSharingAuxLosses
    4 - NoWeightSharing
    5 - NoWeightSharingAuxLosses
"""
NETWORK = 1


if NETWORK==1:
    config_model = dict(
        logs_dir='./logs/SimpleConvNet/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        dataset=SimpleConvNetDataset,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        batch_size=1000,
        num_epochs=200, 
        model=SimpleConvNet(class_num=10, channels_in=1), 
        augmentation=True,
        verbose=1
    )
elif NETWORK==2:
    config_model = dict(
        logs_dir='./logs/WeightSharing/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        dataset=WeightSharingDataset,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        batch_size=1000,
        num_epochs=200, 
        model=WeightSharing(), 
        augmentation=True,
        verbose=1
    )
elif NETWORK==3:
    config_model = dict(
        logs_dir='./logs/WeightSharingAuxLosses/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        dataset=WeightSharingAuxLossesDataset,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        batch_size=1000,
        num_epochs=200,  
        model=WeightSharingAuxLosses(), 
        augmentation=True,
        verbose=1
    )

elif NETWORK==4:
    config_model = dict(
        logs_dir='./logs/NoWeightSharing/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        dataset=NoWeightSharingDataset,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        batch_size=1000,
        num_epochs=200,  
        model=NoWeightSharing(), 
        augmentation=True,
        verbose=1
    )

elif NETWORK==5:
    config_model = dict(
        logs_dir='./logs/NoWeightSharingAuxLosses/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        dataset=NoWeightSharingAuxLossesDataset,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        batch_size=1000,
        num_epochs=200, 
        model=NoWeightSharingAuxLosses(), 
        augmentation=True,
        verbose=1
    )


model, history = train.train(config_model)
