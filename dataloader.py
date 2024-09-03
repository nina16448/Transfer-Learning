import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

batch_size = 16
train_size_rate = 0.8


train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.GaussianBlur(3),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ColorJitter(),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SplitDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

def make_train_dataloader(data_path):
    dataset = datasets.ImageFolder(root=data_path)
    
    train_size = int(len(dataset) * train_size_rate)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_dataset = SplitDataset(train_dataset, transform=train_transforms)
    valid_dataset = SplitDataset(valid_dataset, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def make_test_dataloader(data_path):
    images = []
    for file_name in os.listdir(data_path):
        img_path = os.path.join(data_path, file_name)
        img = Image.open(img_path).convert('RGB')
        img = data_transforms(img)
        images.append(img)
    
    testData = torch.stack(images) 
    test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=4)

    return test_loader