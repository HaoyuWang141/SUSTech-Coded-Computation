import torch
from torch import nn

from decoder.decoder import Decoder
from decoder.sparse_linear import Sparse_linear


class CNNDecoder(Decoder):
    def __init__(
        self, num_in: int, num_out: int, in_dim: tuple, intermediate_channels=32, num_shrink=0
    ):
        super().__init__(num_in, num_out, in_dim)
        int_channels = intermediate_channels * num_in
        s = [1 if num_shrink <= i else 2 for i in range(2)]
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_in * in_dim[0], out_channels=int_channels, kernel_size=3, stride=s[0], padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.08),
            nn.BatchNorm2d(int_channels, eps=0.01),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels, kernel_size=3, stride=s[1], padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.08),
            nn.BatchNorm2d(int_channels, eps=0.01),
        )
        self.block3 = nn.Sequential(
            Sparse_linear(int_channels, int_channels, (in_dim[1] // s[0] // s[1], in_dim[2] // s[0] // s[1])),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.BatchNorm2d(int_channels, eps=0.01),
            nn.Upsample(scale_factor=s[1]),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channels, eps=0.01),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Upsample(scale_factor=s[0]),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channels, eps=0.01),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=int_channels, out_channels=num_out * in_dim[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(datasets) == 0 or self.num_out == 0:
            return []
        datasets = torch.stack(datasets, dim=1)
        batch_size = datasets.size(0)
        datasets = datasets.view(
            batch_size, self.num_in * self.in_dim[0], self.in_dim[1], self.in_dim[2]
        )
        
        out = datasets
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        
        out = out.view(
            batch_size, self.num_out, self.out_dim[0], self.out_dim[1], self.out_dim[2]
        )
        out = list(torch.unbind(out, dim=1))
        return out


if __name__ == "__main__":
    decoder = CNNDecoder(4, 2, (1, 28, 28))
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
