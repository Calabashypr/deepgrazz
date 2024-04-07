from version2.data.mnist_load import MnistDataLoader
from version2.data.rand_data_load import RandomDataLoader
from version2.data.cifar10_load import Cifar10DataLoader

if __name__ == '__main__':
    data_loader = MnistDataLoader()
    # data_loader = RandomDataLoader()
    # data_loader = Cifar10DataLoader()
    data_gen = data_loader.data_gen()
    for i in range(2):
        print('-' * 100)
        data = next(data_gen)
        print(type(data))
        print(data.shape)
        print(data)
