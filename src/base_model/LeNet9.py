import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        return x

    def get_sequence(self):
        # Get sequence of layers except the downsample layer
        return nn.Sequential(
            self.conv,
            self.relu,
            # self.maxpool,
        )

    def children(self):
        # 返回所有非 downsample 属性的子模块
        for name, module in self._modules.items():
            yield module


class LeNet9(BaseModel):

    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(LeNet9, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        in_channels = input_dim[0]
        self.conv_block1 = BasicBlock(in_channels, 32)
        self.conv_block2 = BasicBlock(32, 64)
        self.conv_block3 = BasicBlock(64, 128)
        self.conv_block4 = BasicBlock(128, 256)
        self.conv_block5 = BasicBlock(256, 512)
        self.flatten = nn.Flatten()
        conv_output_size = self.calculate_conv_output(input_dim)
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        self.fc1 = nn.Sequential(nn.Linear(fc_input_size, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        print(x.shape)
        x = self.conv_block2(x)
        print(x.shape)
        x = self.conv_block3(x)
        print(x.shape)
        x = self.conv_block4(x)
        print(x.shape)
        x = self.conv_block5(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

    def get_conv_segment(self) -> nn.Sequential:
        return nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        )

    def get_fc_segment(self) -> nn.Sequential:
        return nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.out,
        )

    def calculate_conv_output(self, input_dim: tuple[int]) -> tuple[int, int, int]:
        # Calculate output dimensions after convolutional layers
        # input_dim: (channels, height, width)
        channels, height, width = input_dim
        out_channels = channels
        for block in self.get_conv_segment():
            for layer in block.children():
                if isinstance(layer, nn.Conv2d):
                    out_channels = layer.out_channels
                    kernel_size = layer.kernel_size
                    stride = layer.stride
                    padding = layer.padding
                    dilation = layer.dilation
                    height = (
                        (
                            height
                            + 2 * padding[0]
                            - dilation[0] * (kernel_size[0] - 1)
                            - 1
                        )
                        // stride[0]
                    ) + 1
                    width = (
                        (
                            width
                            + 2 * padding[1]
                            - dilation[1] * (kernel_size[1] - 1)
                            - 1
                        )
                        // stride[1]
                    ) + 1
                    height, width = int(height), int(width)

        return out_channels, height, width


if __name__ == "__main__":
    # Example usage
    input_dim = (1, 32, 32)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes
    model = LeNet9(input_dim, num_classes)
    # print(model)

    x = torch.randn(1, *input_dim)
    y1 = model(x)
    print(y1.shape)

    # print(model.get_conv_segment())
    conv_segment = model.get_conv_segment()
    y2 = conv_segment(x)
    print(y2.shape)
    fc_segment = model.get_fc_segment()
    y2 = y2.view(y2.size(0), -1)
    y2 = fc_segment(y2)
    print(y2.shape)

    print(y1.data)
    print(y2.data)
