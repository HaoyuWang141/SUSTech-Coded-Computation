import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Sparse_linear(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape, kernel_size=3, stride=1, padding=1):
        super(Sparse_linear, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 获取输入特征图的高度和宽度
        self.input_height, self.input_width = input_shape
        
        # 计算输出特征图的高度和宽度
        self.output_height = (self.input_height - kernel_size + 2 * padding) // stride + 1
        self.output_width = (self.input_width - kernel_size + 2 * padding) // stride + 1
        
        # 为每个输出位置创建独立的卷积核权重和偏置
        self.weights = nn.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.output_height,
                self.output_width,
                kernel_size,
                kernel_size
            )
        )
        init.kaiming_uniform_(self.weights, nonlinearity='relu')
        self.biases = nn.Parameter(torch.zeros(self.out_channels, self.output_height, self.output_width))

    def forward(self, x):
        batch_size = x.size(0)
        
        # 在提取 patch 之前进行 padding
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # 初始化输出张量
        output = torch.zeros((batch_size, self.out_channels, self.output_height, self.output_width), device=x.device)
        
        # 遍历输出特征图的每个位置进行卷积操作
        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # 从填充后的输入张量中提取当前位置的 patch
                x_patch = x_padded[:, :, h_start:h_end, w_start:w_end]
                w_patch = self.weights[:, :, i, j, :, :].contiguous()
                b_patch = self.biases[:, i, j].contiguous()
                
                # 对当前 patch 应用特定位置的卷积核和偏置
                out_patch = F.conv2d(
                    x_patch,
                    w_patch,
                    bias=b_patch,
                    stride=1,
                    padding=0
                )
                output[:, :, i, j] = out_patch.squeeze(2).squeeze(2)
        
        return output


if __name__ == '__main__':
    # 示例用法
    input_tensor = torch.randn(1, 3, 32, 32)  # 批量大小为1，3个输入通道，32x32的输入尺寸
    conv_layer = Sparse_linear(in_channels=3, out_channels=16, input_shape=(32, 32), kernel_size=3, stride=1, padding=1)
    output_tensor = conv_layer(input_tensor)
    print(output_tensor.shape)  # 应输出 torch.Size([1, 16, 32, 32])
