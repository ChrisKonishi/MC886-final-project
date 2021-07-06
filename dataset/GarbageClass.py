
# Precisa definir como saber a quantidade de itens e como conseguir um, precisa implementar
# __len__ = len(data)
# __index__ = data[idx]
# Os dados podem ou não ficar todo em memória, se der, coloca tudo na ram
class GarbageClass():
    def __init__(self, mode='train'): #mode: train, val, test
        #load dataset
        pass

    def __len__(self):
        # Quantity of Items
        pass

    def __index__(self, idx):
        # return an item (img, label)
        pass