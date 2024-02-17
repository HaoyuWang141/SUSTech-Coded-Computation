import torch
from torch import nn

from decoder.decoder import Decoder


class CatChannelConvDecoder(Decoder):
    def __init__(
        self, num_in: int, num_out: int, in_dim: tuple, intermediate_channels=20
    ):
        super().__init__(num_in, num_out, in_dim)
        int_channels = intermediate_channels * num_in
        self.nn = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in * in_dim[0],
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=num_out * in_dim[0],
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(datasets) == 0 or self.num_out == 0:
            return []
        datasets = torch.stack(datasets, dim=1)
        batch_size = datasets.size(0)
        datasets = datasets.view(
            batch_size, self.num_in * self.in_dim[0], self.in_dim[1], self.in_dim[2]
        )
        out = self.nn(datasets)
        out = out.view(
            batch_size, self.num_out, self.out_dim[0], self.out_dim[1], self.out_dim[2]
        )
        out = list(torch.unbind(out, dim=1))
        return out


class CatBatchSizeConvDecoder(Decoder):
    def __init__(
        self, num_in: int, num_out: int, in_dim: tuple, intermediate_channels=20
    ):
        super().__init__(num_in, num_out, in_dim)
        int_channels = intermediate_channels * num_in
        self.nn = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=int_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int_channels,
                out_channels=num_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        datasets = torch.stack(datasets, dim=1)
        # TODO: datasets 维度为 (batch_size, k, c, h, w)，如何以论文的方式缩减为(new_batch_size, k, h, w)输入网络，再转换输出？
        return None


if __name__ == "__main__":
    decoder = CatChannelConvDecoder(4, 2, (1, 28, 28))
    print(decoder)
    sample_data = [
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
    ]
    output = decoder(sample_data)
    print(f"output num: {len(output)}")
    print(f"output shape: {output[0].shape}")
