"""Dataset Class"""
import os
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

class CIFAR10_(CIFAR10):
    """
    CIFAR10 Dataset
    """

    img_size = (3, 32, 32)

    def __init__(
        self,
        root,
        split="training",
        transform=ToTensor(), 
        target_transform=None, 
        download=False
    ):
        if split in ["training", "validation"]:
            train = True
        elif split == 'test':
            train = False
        else:
            ValueError
        super(CIFAR10_, self).__init__(
            root, 
            train, 
            transform=transform, 
            target_transform=target_transform, 
            download=download)

        if split == 'training':
            self.data = self.data[:45000]
            self.targets = self.targets[:45000]
        elif split == 'validation':
            self.data = self.data[45000:]
            self.targets = self.targets[45000:]
        
        print(f"Split {split} | {self.data.shape}")