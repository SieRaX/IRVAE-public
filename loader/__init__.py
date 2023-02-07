from torch.utils import data

from loader.MNIST_dataset import MNIST_
from loader.CIFAR_dataset import CIFAR10_

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'MNIST':
        dataset = MNIST_(**data_dict)
    elif name == 'CIFAR10':
        dataset = CIFAR10_(**data_dict)
    return dataset