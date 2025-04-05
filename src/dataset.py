import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(
                        data_dir,
                        batch_size,
                        augment,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True
                    ):
    """
    Creates data loaders for training and validation from the CIFAR-10 dataset.

    This function returns DataLoader objects for training and validation sets, with options
    for applying data augmentation and splitting the dataset based on a given validation size.
    The training dataset is augmented based on the provided argument, while the validation 
    dataset is standardized.

    Args:
        data_dir (str): Path to the directory where CIFAR-10 dataset will be stored.
        batch_size (int): Number of samples per batch to load.
        augment (bool): Whether to apply data augmentation on the training dataset.
        random_seed (int): Seed for shuffling and random number generation, ensuring 
            reproducibility.
        valid_size (float): Fraction of the training data to be used for validation (between 0 and 1).
            Defaults to 0.1 (10% of the data).
        shuffle (bool): Whether to shuffle the training dataset before splitting into 
            training and validation sets. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.

    Transforms:
        - For training (if augment is True): Random cropping and horizontal flip are applied, 
        followed by resizing to 227x227, normalization, and converting the image to a tensor.
        - For validation: Resizing to 227x227, normalization, and converting the image to a tensor.

    The data is shuffled using the specified random seed to ensure reproducibility, and the
    dataset is split into training and validation sets based on the `valid_size` parameter.
    """

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # define transforms
    valid_transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), normalize])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])
    
    # load the dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    """
    Creates a data loader for the test dataset from the CIFAR-10 dataset.

    This function returns a DataLoader object for the test set, with the option to shuffle 
    the data. The test dataset is standardized using the given normalization values and resized 
    to 227x227 pixels.

    Args:
        data_dir (str): Path to the directory where CIFAR-10 dataset will be stored.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Whether to shuffle the test dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the test set.

    Transforms:
        - The images are resized to 227x227, normalized using the ImageNet normalization values
        (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), and converted to a tensor.

    The test dataset is loaded without any data augmentation, and the specified batch size is 
    used for efficient testing.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader
