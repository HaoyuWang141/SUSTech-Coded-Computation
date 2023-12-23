import torch
import torch.nn as nn
from base_model.BaseModel import BaseModel
import torch.nn.init as init


class VGG16(BaseModel):
    def __init__(self, input_dim: tuple[int], num_classes: int) -> None:
        super(VGG16, self).__init__()
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
        fc_input_size = conv_output_size[0] * conv_output_size[1] * conv_output_size[2]
        self.fc_block = nn.Sequential(
            nn.Linear(fc_input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
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
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        # x = torch.softmax(x, dim=1)
        return x

    def get_conv_segment(self) -> nn.Sequential:
        return nn.Sequential(self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5)

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

        conv1_1 = conv2d_out_size((height, width))
        conv1_2 = conv2d_out_size(conv1_1)
        pool1 = maxpool2d_out_size(conv1_2)

        conv2_1 = conv2d_out_size(pool1)
        conv2_2 = conv2d_out_size(conv2_1)
        pool2 = maxpool2d_out_size(conv2_2)

        conv3_1 = conv2d_out_size(pool2)
        conv3_2 = conv2d_out_size(conv3_1)
        conv3_3 = conv2d_out_size(conv3_2)
        pool3 = maxpool2d_out_size(conv3_3)

        conv4_1 = conv2d_out_size(pool3)
        conv4_2 = conv2d_out_size(conv4_1)
        conv4_3 = conv2d_out_size(conv4_2)
        pool4 = maxpool2d_out_size(conv4_3)

        conv5_1 = conv2d_out_size(pool4)
        conv5_2 = conv2d_out_size(conv5_1)
        conv5_3 = conv2d_out_size(conv5_2)
        pool5 = maxpool2d_out_size(conv5_3)

        return (
            512,  # 512 is the number of channels after conv5_3
            pool5[0],  # =7
            pool5[1],  # =7
        )


if __name__ == "__main__":
    # Example usage
    input_dim = (3, 224, 224)  # Example input dimensions (channels, height, width)
    num_classes = 10  # Example number of output classes
    model = VGG16(input_dim, num_classes)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

    print(model.get_conv_segment())
    conv_segment = model.get_conv_segment()
    y = conv_segment(x)
    print(y.shape)

