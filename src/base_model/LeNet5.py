import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel


class LeNet5(BaseModel):
    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(LeNet5, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        conv_output_size = self.calculate_conv_output(input_dim)
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_size, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

    def get_conv_segment(self) -> nn.Sequential:
        layers = []
        for layer in self.conv_block1:
            layers.append(layer)
        for layer in self.conv_block2:
            layers.append(layer)

        conv_segment = nn.Sequential(*layers)
        return conv_segment

    def get_fc_segment(self) -> nn.Sequential:
        layers = []
        for layer in self.fc1:
            layers.append(layer)
        for layer in self.fc2:
            layers.append(layer)
        layers.append(self.out)
        fc_segment = nn.Sequential(*layers)
        return fc_segment

    def calculate_conv_output(self, input_dim: tuple[int]) -> tuple[int, int, int]:
        # Assuming input_dim is a tuple (channels, height, width)
        channels, height, width = input_dim

        def conv2d_out_size(size, kernel_size=5, stride=1, padding=0):
            return (
                (size[0] - kernel_size - 2 * padding) // stride + 1,
                (size[1] - kernel_size - 2 * padding) // stride + 1,
            )

        def maxpool2d_out_size(size, kernel_size=2, stride=2, padding=0):
            return (
                (size[0] - kernel_size - 2 * padding) // stride + 1,
                (size[1] - kernel_size - 2 * padding) // stride + 1,
            )

        conv1_out = conv2d_out_size((height, width))  # Conv1
        maxpool1_out = maxpool2d_out_size(conv1_out)  # MaxPool
        conv2_out = conv2d_out_size(maxpool1_out)  # Conv2
        maxpool2_out = maxpool2d_out_size(conv2_out)  # MaxPool
        return (
            16,
            maxpool2_out[0],
            maxpool2_out[1],
        )  # 16 is the number of channels after conv2


if __name__ == "__main__":
    # Example usage
    input_dim = (1, 28, 28)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes
    model = LeNet5(input_dim, num_classes)
    print(model)

    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

    print(model.get_conv_segment())
    conv_segment = model.get_conv_segment()
    y = conv_segment(x)
    print(y.shape)
