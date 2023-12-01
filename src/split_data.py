from torchvision import datasets, transforms
from util.split_data import split_data
from base_model.LeNet5 import LeNet5
import torch


model = LeNet5(input_dim=(1, 28, 28), num_classes=10)
base_model_path = "base_model/LeNet5/MNIST/2023_11_28/model.pth"
model.load_state_dict(torch.load(base_model_path))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
print(f"Test dataset: {len(dataset)}")
print("image size: ", dataset[0][0].size())

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=1,
#     train=False,
#     output_file=f"./data/MNIST/split/1/split_test_datasets.pt",
# )

# print("-" * 50)

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=2,
#     train=False,
#     output_file=f"./data/MNIST/split/2/split_test_datasets.pt",
# )

# print("-" * 50)

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=4,
#     train=False,
#     output_file=f"./data/MNIST/split/4/split_test_datasets.pt",
# )

# print("-" * 50)

# dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# print(f"Train dataset: {len(dataset)}")
# print("image size: ", dataset[0][0].size())

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=1,
#     train=True,
#     output_file=f"./data/MNIST/split/1/split_train_datasets.pt",
# )

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=2,
#     train=True,
#     output_file=f"./data/MNIST/split/2/split_train_datasets.pt",
# )

# print("-" * 50)

# split_data(
#     layers=model.get_conv_segment(),
#     dataset=dataset,
#     split_num=4,
#     train=True,
#     output_file=f"./data/MNIST/split/4/split_train_datasets.pt",
# )
