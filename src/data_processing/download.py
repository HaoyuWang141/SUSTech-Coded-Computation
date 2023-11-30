# download datasets

import torchvision
from torchvision import datasets, transforms

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Downloading the MNIST training dataset
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Downloading the MNIST testing dataset
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
