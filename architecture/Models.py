import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet101, inception_v3

def Resnet18(n_class):
    model = resnet18()
    model.fc = nn.Linear(512, n_class)
    return model

def Resnet101(n_class):
    model = resnet101()
    model.fc = nn.Linear(2048, n_class)
    return model

def InceptionV3(n_class):
    model = inception_v3()
    model.fc = nn.Linear(2048, n_class)
    return model

if __name__ == '__main__':
    print(InceptionV3(5))
    pass