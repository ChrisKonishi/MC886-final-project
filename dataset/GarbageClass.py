from torch.utils.data import Dataset

# Precisa definir como saber a quantidade de itens e como conseguir um, precisa implementar
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# __len__ = len(data)
# __index__ = data[idx]
# Os dados podem ou não ficar todo em memória, se der, coloca tudo na ram
# Data augmentation é feito aqui, vide transforms. Precisa redimensionar a imagem para o tamnho certo tbm
#   https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class GarbageClass(Dataset):
    def __init__(self, mode='train', device='cpu'): #mode: train, val, test. device: 'cpu' ou 'cuda' a = Tensor.. a.to(device)
        #load dataset
        pass

    def __len__(self):
        # Quantity of Items
        pass

    def __getitem__(self, idx):
        # return an item (img, label) img as normalized tenseor pls
        pass