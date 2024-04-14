import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel
import torch.nn.init as init


class VGG10(BaseModel):
    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(VGG10, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
        )
        conv_output_size = self.calculate_conv_output(input_dim)
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        print(f"fc_input_size: {fc_input_size}")
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        # print(x.shape)
        x = self.conv_block2(x)
        # print(x.shape)
        x = self.conv_block3(x)
        # print(x.shape)
        x = self.conv_block4(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.fc_block(x)
        # x = torch.softmax(x, dim=1)
        return x

    def get_conv_segment(self) -> nn.Sequential:
        return nn.Sequential(*self.conv_block1, *self.conv_block2, *self.conv_block3, *self.conv_block4)

    def get_fc_segment(self) -> nn.Sequential:
        return self.fc_block

    def calculate_conv_output(self, input_dim: tuple[int]) -> tuple[int, int, int]:
        # Assuming input_dim is a tuple (channels, height, width)
        channels, height, width = input_dim

        def conv2d_out_size(size, kernel_size=3, stride=1, padding=1):
            return (
                (size[0] - kernel_size + 2 * padding) // stride + 1,
                (size[1] - kernel_size + 2 * padding) // stride + 1,
            )

        def maxpool2d_out_size(size, kernel_size=2, stride=2, padding=0):
            return (
                (size[0] - kernel_size + 2 * padding) // stride + 1,
                (size[1] - kernel_size + 2 * padding) // stride + 1,
            )

        for layer in self.get_conv_segment():
            if isinstance(layer, nn.Conv2d):
                padding = layer.padding[0]
                channels = layer.out_channels
                height, width = conv2d_out_size((height, width), padding=padding)
            elif isinstance(layer, nn.MaxPool2d):
                height, width = maxpool2d_out_size((height, width))
                
        return channels, height, width

if __name__ == "__main__":
    # Example usage
    input_dim = (3, 32, 32)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes
    model = VGG10(input_dim, num_classes)

    x = torch.randn(1, *input_dim)
    y1 = model(x)
    print(y1.shape)

    conv_segment = model.get_conv_segment()
    y2 = conv_segment(x)
    print(y2.shape)
    y2 = y2.view(y2.size(0), -1)
    fc_segment = model.get_fc_segment()
    y2 = fc_segment(y2)
    print(y2.shape)
    
    print(y1.data)
    print(y2.data)
