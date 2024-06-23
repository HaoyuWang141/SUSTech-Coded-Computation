import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel
import torch.nn.init as init


class Alexnet(BaseModel):
    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(Alexnet, self).__init__()
        input_dim = tuple(input_dim)
        num_classes = int(num_classes)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2)),  # output[128, 13, 13]
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        conv_output_size = self.calculate_conv_output(input_dim)
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        self.fc_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def get_conv_segment(self) -> nn.Sequential:
        return nn.Sequential(self.conv_block1, self.conv_block2, self.conv_block3)

    def get_fc_segment(self) -> nn.Sequential:
        return self.fc_block

    def calculate_conv_output(self, input_dim: tuple[int, int, int]) -> tuple[int, int, int]:
        C, H, W = input_dim
        # Conv1
        H = (H + 2*2 - 11) // 4 + 1
        W = (W + 2*2 - 11) // 4 + 1
        H = (H - 3) // 2 + 1
        W = (W - 3) // 2 + 1
        # Conv2
        H = (H + 2*2 - 5) // 1 + 1
        W = (W + 2*2 - 5) // 1 + 1
        H = (H - 3) // 2 + 1
        W = (W - 3) // 2 + 1
        # Conv3-1, Conv3-2, Conv3-3
        H = (H + 2*1 - 3) // 1 + 1
        W = (W + 2*1 - 3) // 1 + 1
        H = (H + 2*1 - 3) // 1 + 1
        W = (W + 2*1 - 3) // 1 + 1
        H = (H + 2*1 - 3) // 1 + 1
        W = (W + 2*1 - 3) // 1 + 1
        # MaxPool
        H = (H - 3) // 2 + 1
        W = (W - 3) // 2 + 1
        # 输出通道数在最后一个卷积层后变为128
        return (128, H, W)

