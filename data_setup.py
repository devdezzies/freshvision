"""
contains functionality for creating pytorch dataloaders for image classification data
"""
import os 
import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 
from pathlib import Path 
import pathlib
import requests
import zipfile
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
from PIL import Image

NUM_WORKERS = os.cpu_count()

# create custom dataset
def find_classes(directory: str) -> Tuple[list[str], Dict[str, int]]:
    """
    Finds the class folder names in a target directory 
    """
    # 1. get the class names by scanning the target directory 
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. raise an error is class names couldn't be found 
    if not classes:
        raise FileNotFoundError(f"couldn't find any classes in {directory}")
    
    # 3. create a dictionary of index labels 
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# 1. subclass torch.utils.data.Dataset 
class ImageFolderCustom(Dataset):
    # 2. initialize the constructor
    def __init__(self, targ_dir: str, heads: list[str], transform=None, is_training: bool = True):
        # 3. create several attributes 
        # get all the image paths
        self.training = []
        self.testing = []
        for tag in heads: 
            self.img_list = list(Path(targ_dir / tag).glob("*.jpg"))
            self.train_length = int(len(self.img_list) * 0.8)
            self.training.extend(self.img_list[:self.train_length])
            self.testing.extend(self.img_list[self.train_length:])

        if is_training: 
            self.paths = self.training
        else: 
            self.paths = self.testing
        # setup transforms
        self.transform = transform
        # create classes and class_to_idx 
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. create a function to load images 
    def load_image(self, index: int) -> Image.Image: 
        "opens an image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # 5. overwrite __len__()
    def __len__(self) -> int: 
        return len(self.paths)
    
    # 6. overwrite __getitem__() to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "returns one sample of data, data and the label (X, y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # transform if necessary 
        if self.transform:
            return self.transform(img), class_idx
        else: 
            return img, class_idx

def create_dataloaders(
    image_dir: str,  
    heads: list[str],
    train_transform: transforms.Compose, 
    test_transform: transforms.Compose,
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    """
    creates training and testing DataLoaders. 

    Takes in a training directory and testing directory path and turns them
    into pytorch datasets and then into pytorch dataloaders. 

    Args:
        train_dir: path to training directory. 
        test_dir: path to testing directory 
        transform: torchvision transforms to perform on training and testing data. 
        batch_size: number of samples per batch in each of the dataloaders. 
        num_workers: an integer for number of workers per dataloader.

    returns: 
        A tuple of (train_dataloader, test_dataloader, class_names).
        where class_names is a list of the target classes. 

        Example usage: 
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir, 
                                                                                test_dir=path/to/test_dir, 
                                                                                transform=some_transform,
                                                                                batch_size=32,
                                                                                num_workers=4)                                                                                                                 
    """

    # use ImageFolder to create datasets 
    train_data = ImageFolderCustom(targ_dir=image_dir, heads=heads, transform=train_transform, is_training=True)

    test_data = ImageFolderCustom(targ_dir=image_dir, heads=heads, transform=test_transform, is_training=False)

    # get class names 
    class_names = train_data.classes 

    # turn images into dataloaders 
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
