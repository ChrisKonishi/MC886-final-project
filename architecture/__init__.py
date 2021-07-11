from .Models import Resnet18, Resnet101, InceptionV3

models = {
    'resnet-18': (Resnet18, (224,224))
    , 'resnet-101': (Resnet101, (224,224))
    , 'inception-v3': (InceptionV3, (299, 299))
}
