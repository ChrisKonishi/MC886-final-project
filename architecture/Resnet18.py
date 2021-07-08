import torch
import torch.nn as nn
from torchvision.models import resnet18

#Não precisa fazer isso, pode ser um wrapper para a Resnet do Pytorch (https://pytorch.org/vision/stable/models.html)
def Resnet18(n_class):
    model = resnet18()
    model.fc = nn.Linear(512, n_class)
    return model

if __name__ == '__main__':
    print(Resnet18(5))
    pass