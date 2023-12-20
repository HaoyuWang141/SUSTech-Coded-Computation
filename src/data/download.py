# download datasets

import torchvision
from torchvision import datasets, transforms

# MINIST

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Downloading the MNIST training dataset
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Downloading the MNIST testing dataset
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# CIFAR10

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Downloading the CIFAR10 training dataset
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Downloading the CIFAR10 testing dataset
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)