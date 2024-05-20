import torch
from torch import nn
from math import prod

from encoder.encoder import Encoder


class MLPEncoder(Encoder):
    def __init__(self, num_in: int, num_out: int, in_dim: tuple):
        super().__init__(num_in, num_out, in_dim)
        self.nn = nn.Sequential(
            nn.Linear(num_in, num_in),
            nn.ReLU(),
            nn.Linear(num_in, num_out),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(datasets) == 0 or self.num_out == 0:
            return []
        datasets = torch.stack(datasets, dim = -1) # 把K个图片 按照最后一个维度进行堆叠
        original_batch_size = datasets.size(0) # batch_size
        batch_size = datasets.size(0) * datasets.size(1) * datasets.size(2) * datasets.size(3) # 将batch_size以及channel height width进行合并，都转换为batch_size
        datasets = datasets.view(batch_size, -1)
        out = self.nn(datasets)
        out = out.view(original_batch_size, *self.out_dim, self.num_out)
        out = list(torch.unbind(out, dim = -1))
        return out


if __name__ == "__main__":
    encoder = MLPEncoder(4, 2, (1, 28, 28))
    print(encoder)
    sample_data = [
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
        torch.randn(5, 1, 28, 28),
    ]
    output = encoder(sample_data)
    print(f"output num: {len(output)}")
    print(f"output shape: {output[0].shape}")
