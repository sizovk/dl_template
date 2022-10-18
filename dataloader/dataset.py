from torchvision import datasets, transforms

class MNIST(datasets.MNIST):

    def __init__(self, data_dir):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super().__init__(data_dir, train=True, download=True, transform=trsfm)
