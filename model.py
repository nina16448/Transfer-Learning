import torch.nn as nn
from torchvision import models

def MyResNet50():
    model = models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    num_classes = 12
    
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model