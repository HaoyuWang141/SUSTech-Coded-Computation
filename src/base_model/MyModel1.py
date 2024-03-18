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

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def get_sequence(self):
        # Get sequence of layers except the downsample layer
        return nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
        )

    def children(self):
        # 返回所有非 downsample 属性的子模块
        for name, module in self._modules.items():
            yield module


class MyModel1(BaseModel):

    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(MyModel1, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        in_channels = input_dim[0]
        self.conv_block1 = BasicBlock(in_channels, 6)
        self.conv_block2 = BasicBlock(6, 8)
        self.conv_block3 = BasicBlock(8, 10)
        self.conv_block4 = BasicBlock(10, 12)
        conv_output_size = self.calculate_conv_output(input_dim)
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        self.fc1 = nn.Sequential(nn.Linear(fc_input_size, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
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
                        / stride[0]
                    ) + 1
                    width = (
                        (
                            width
                            + 2 * padding[1]
                            - dilation[1] * (kernel_size[1] - 1)
                            - 1
                        )
                        / stride[1]
                    ) + 1
                    height, width = int(height), int(width)
        return out_channels, height, width


if __name__ == "__main__":
    # Example usage
    input_dim = (1, 28, 28)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes
    model = MyModel1(input_dim, num_classes)
    # print(model)

    x = torch.randn(1, 1, 28, 28)
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
