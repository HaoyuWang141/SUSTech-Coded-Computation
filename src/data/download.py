# download datasets

import torchvision
from torchvision import datasets, transforms

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# MINIST
# input_shape = (1, 28, 28), class_num = 10

# Downloading the MNIST training dataset
train_set = datasets.MNIST(root="./", train=True, download=True, transform=transform)

# Downloading the MNIST testing dataset
test_set = datasets.MNIST(root="./", train=False, download=True, transform=transform)


# CIFAR10
# input_shape = (3, 32, 32), class_num = 10

# Downloading the CIFAR10 training dataset
train_set = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)

# Downloading the CIFAR10 testing dataset
test_set = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)


# CIFAR100
# input_shape = (3, 32, 32), class_num = 100

# Downloading the CIFAR100 training dataset
train_set = datasets.CIFAR100(root="./", train=True, download=True, transform=transform)

# Downloading the CIFAR100 testing dataset
test_set = datasets.CIFAR100(root="./", train=False, download=True, transform=transform)
