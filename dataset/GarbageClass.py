from torch.utils.data import Dataset
import pandas as pd
import os, os.path as osp
import re
from PIL import Image
import torchvision.transforms as transforms

try:
    from .CustomTransforms import GaussianNoise
except ImportError:
    from CustomTransforms import GaussianNoise


# Precisa definir como saber a quantidade de itens e como conseguir um, precisa implementar
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# __len__ = len(data)
# __index__ = data[idx]
# Os dados podem ou não ficar todo em memória, se der, coloca tudo na ram
# Data augmentation é feito aqui, vide transforms. Precisa redimensionar a imagem para o tamnho certo tbm
#   https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class GarbageClass(Dataset):
    def __init__(self, mode='train', size=(224,224)): #mode: train, val, test. size = (h,w)
        self.mode = mode
        self.size = size

        self.dir = osp.join(osp.dirname(osp.realpath(__file__)), 'garbage_classification/')
        self.imgs = []
        self.labels = []

        #definir os transform https://pytorch.org/vision/stable/transforms.html (da para criar personalizado tambem, eh facil)
        if mode == 'train':
            idx_file = self.dir+'one-indexed-files-notrash_train.txt'
            compose = [
                transforms.RandomResizedCrop(size, scale=(0.8, 1))
                , transforms.RandomHorizontalFlip()
                , transforms.RandomVerticalFlip()
                , GaussianNoise(0.02)
            ]
        elif mode == 'val':
            idx_file = self.dir+'one-indexed-files-notrash_val.txt'
        elif mode == 'test':
            idx_file = self.dir+'one-indexed-files-notrash_test.txt'
        else:
            raise Exception(f'Invalid mode: {mode}')
        if mode in ['val', 'test']:
            compose = [
                transforms.Resize(size)
            ]
        compose.append(transforms.ToTensor())
        self.transform = transforms.Compose(compose)

        self.index = pd.read_csv(idx_file, header=None, names=['img', 'class'], sep='\s+')
        self._load_files()

    def _load_files(self):
        material_re = re.compile(r'[a-zA-Z]+')
        for idx, row in self.index.iterrows():
            file, label = row['img'], row['class']
            material = material_re.match(file).group()
            im = Image.open(osp.join(self.dir, material, file))
            if im:
                self.imgs.append(im)
                self.labels.append(label)
            else:
                continue

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
           img = self.transform(img)
        return (img, self.labels[idx])

if __name__ == '__main__': #test
    data = GarbageClass(mode='test')
    print('len:', len(data))
    transforms.functional.to_pil_image(data[100][0]).show()