import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import os

from dataset.splited_dataset import (
    SplitedTrainDataset,
    SplitedTestDataset,
)


def split_data(
    layers: nn.Sequential,
    dataset: VisionDataset,
    split_num: int,
    train: bool = False,
    output_file: str = None,
) -> list[Dataset]:
    """
    Partition input data from a VisionDataset into partition_num partitions.

    :param layers: the conv segment of the model.
    :param dataset: the dataset object containing image data. [(image, label), ...)], image.shape = (channel, height, width)
    :param partition_num: number of partitions.

    :return: partitioned input data. [VisionDataset, ...], VisionDataset = [(image, label), ...)], image.shape = (channel, height, width(partial))
    """
    input_size = dataset[0][0].size()  # (channel, height, width)
    output_size = cal_output_size(layers, input_size)  # (channel, height, width)
    output_channel, output_height, output_width = output_size

    images_list = []
    conv_segment_labels = []

    for output_range in partition_range(output_height, output_width, split_num):
        # [start, end)
        input_range = cal_input_range(
            layers, output_range
        )  # ((start_h, start_w), (end_h, end_w))

        images = []

        dataset_tqdm = tqdm(
            dataset, desc=f"Spliting dataset, input_range is {input_range}"
        )
        for img, _ in dataset_tqdm:
            start, end = input_range
            start_h, start_w = start
            end_h, end_w = end
            images.append(img[:, start_h:end_h, start_w:end_w])

            if train:
                img = img.to(layers[0].weight.device)
                conv_segment_labels.append(layers(img.unsqueeze(0)).squeeze(0))

        images_list.append(torch.stack(images, dim=0))

    if train:
        labels = conv_segment_labels
        splited_dataset = SplitedTrainDataset(
            images_list=images_list,
            labels=labels,
        )
    else:
        labels = [label for _, label in dataset]
        splited_dataset = SplitedTestDataset(
            images_list=images_list,
            labels=labels,
        )

    if output_file is not None:
        path = os.path.dirname(output_file)
        if not os.path.exists(path):
            os.makedirs(path)
        splited_dataset.save(output_file)

    return splited_dataset


def cal_output_size(layers: nn.Sequential, input_size: tuple) -> tuple:
    """
    Calculate the output range of the conv segment

    :param layers: the conv segment of model
    :param input_range: the input range of the conv segment, (channel, height, width)

    :return: the output range of the conv segment, (channel, height, width)
    """
    x = torch.zeros(1, *input_size)
    device = layers[0].weight.device
    x = x.to(device)
    y = layers(x)
    return tuple(y.size()[1:])


def cal_input_range(layers: nn.Sequential, output_range: tuple) -> tuple:
    """
    Calculate the input range of the conv segment

    :param layers: the conv segment of model
    :param output_range: the output range of the conv segment, ((start_h, start_w), (end_h, end_w))

    :return: the input range of the conv segment
    """
    # output_range is ((start_height, start_width), (end_height, end_width))
    start_h, start_w = output_range[0]
    end_h, end_w = output_range[1]

    for layer in reversed(layers):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            kernel_size = (
                layer.kernel_size
                if isinstance(layer.kernel_size, tuple)
                else (layer.kernel_size, layer.kernel_size)
            )
            stride = (
                layer.stride
                if isinstance(layer.stride, tuple)
                else (layer.stride, layer.stride)
            )
            padding = (
                layer.padding
                if isinstance(layer.padding, tuple)
                else (layer.padding, layer.padding)
            )

            # 反向计算高度范围
            start_h = max(0, (start_h * stride[0]) - padding[0])
            end_h = (end_h - 1) * stride[0] + kernel_size[0] - padding[0]

            # 反向计算宽度范围
            start_w = max(0, (start_w * stride[1]) - padding[1])
            end_w = (end_w - 1) * stride[1] + kernel_size[1] - padding[1]

    return ((start_h, start_w), (end_h, end_w))


def partition_range(height, width, partition_num):
    return partition_strategy1(height, width, partition_num)

"""
partition strategy 1:
    partition the width of the image into partition_num partitions
    
TODO: support more partition strategies, e.g. partition height, partition both height and width
"""

def partition_strategy1(height, width, partition_num):
    assert width % partition_num == 0
    partitioned_width = width // partition_num
    for i in range(partition_num):
        yield (
            (0, partitioned_width * i),
            (height, partitioned_width * (i + 1)),
        )


if __name__ == "__main__":
    # Example usage
    input_dim = (1, 28, 28)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes

    from base_model.LeNet5 import LeNet5

    model = LeNet5(input_dim, num_classes)

    conv_segment = model.get_conv_segment()
    print(conv_segment)

    print("-" * 20)

    print("test cal_output_size():")
    x = torch.randn(1, 1, 28, 28)
    y = conv_segment(x)
    print(y.size())

    output_size = cal_output_size(conv_segment, input_dim)
    print(output_size)

    print("-" * 20)

    print("test partition_range():")
    channel, height, width = output_size
    output_range_list = []
    for output in partition_range(height, width, 4):
        output_range_list.append(output)
        print(output)

    print("-" * 20)

    print("test cal_input_range():")
    input_range_list = []
    for output_range in output_range_list:
        input_range = cal_input_range(conv_segment, output_range)
        input_range_list.append(input_range)
        print(input_range)

    print("-" * 20)

    print("test split_data():")
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    print(f"Test dataset: {len(dataset)}")
    print("image size: ", dataset[0][0].size())
    K = 2
    split_data(
        conv_segment,
        dataset,
        K,
        train=False,
    )

    print("data_partition() test passed!")
