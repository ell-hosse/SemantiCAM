import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from config import NUM_CLASSES

def get_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
