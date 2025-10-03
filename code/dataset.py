from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 128

train_data = MNIST(root="data/", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST(root="data/", train=False, download=True, transform=transforms.ToTensor())


train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)


adversarial_test_loader = DataLoader(test_data, batch_size=1, shuffle=False)