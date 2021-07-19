import random
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def make_dataset_grid(data, save=True):
    ind = random.sample(range(0,len(data)), 64)
    imgs = [data[i][0] for i in ind]
    imgs = np.transpose(make_grid(imgs, normalize=True), (1,2,0))
    if save:
        plt.imsave('./img_grid.pdf', imgs)
    return imgs


if __name__ == '__main__':
    from os.path import realpath, dirname, join
    import sys
    sys.path.append(join(dirname(dirname(realpath(__file__))), 'dataset'))
    from GarbageClass import GarbageClass

    data = GarbageClass(mode='train')
    make_dataset_grid(data)
