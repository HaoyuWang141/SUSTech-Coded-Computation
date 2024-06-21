import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel
import torch.nn.init as init
from typing import List


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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        conv_output_size = self.calculate_conv_output(input_dim)
        print(f"conv_output_size: {conv_output_size}")
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

    def forward(self, x: torch.Tensor, models: List[nn.Module]) -> torch.Tensor:
        for model in models:
            x = model(x)
        return x

    def get_conv_segment(self, index: int) -> nn.Sequential:
        block = [None]
        if index == 1:
            block = self.conv_block1
            block = list(block.children())  # Convert to list
            block.pop()  # Remove the last layer
            block = nn.Sequential(*block)  # Convert back to nn.Sequential
        elif index == 2:
            block = self.conv_block2
            block = list(block.children())
            block.pop()
            block = nn.Sequential(*block)
        elif index == 3:
            block = self.conv_block3
            block = list(block.children())
            block.pop()
            block = nn.Sequential(*block)
        elif index == 4:
            block = self.conv_block4
            block = list(block.children())
            block.pop()
            block = nn.Sequential(*block)
        elif index == 5:
            block = self.conv_block5
            block = list(block.children())
            block.pop()
            block = nn.Sequential(*block)
        return block
    

    def get_flatten(self) -> nn.Sequential:
        return self.flatten
    
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

        for i in range(5) :
            for layer in self.get_conv_segment(index = i + 1):
                if isinstance(layer, nn.Conv2d):
                    padding = layer.padding[0]
                    channels = layer.out_channels
                    height, width = conv2d_out_size((height, width), padding=padding) 
            height, width = maxpool2d_out_size((height, width))
                
        return channels, height, width

