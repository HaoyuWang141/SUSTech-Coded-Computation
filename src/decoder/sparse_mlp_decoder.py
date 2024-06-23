import torch
from torch import nn

from decoder.decoder import Decoder
from decoder.sparse_linear import Sparse_linear


class SparseMLPDecoder(Decoder):
    def __init__(self, num_in: int, num_out: int, in_dim: tuple, intermediate_channels=20) -> None:
        super().__init__(num_in, num_out, in_dim)
        int_channels = intermediate_channels * num_in
        self.nn = nn.Sequential(
            Sparse_linear(num_in * in_dim[0], int_channels, in_dim[1:3]),
            nn.ReLU(),
            Sparse_linear(int_channels, int_channels, in_dim[1:3]),
            nn.ReLU(),
            Sparse_linear(int_channels, num_out * in_dim[0], in_dim[1:3]),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        # if len(datasets) == 0 or self.num_out == 0:
        #     return []
        # datasets = torch.stack(datasets, dim = -1)
        # original_batch_size = datasets.size(0)
        # batch_size = datasets.size(0) * datasets.size(1) * datasets.size(2) * datasets.size(3)
        # datasets = datasets.view(batch_size, -1)
        # out = self.nn(datasets)
        # out = out.view(original_batch_size, *self.out_dim, self.num_out)
        # out = list(torch.unbind(out, dim = -1))
        # return out
    
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


if __name__ == "__main__":
    decoder = SparseMLPDecoder(6, 4, (16, 4, 1))
    print(decoder)
    sample_data = [
        torch.randn(5, 16, 4, 1),
        torch.randn(5, 16, 4, 1),
        torch.randn(5, 16, 4, 1),
        torch.randn(5, 16, 4, 1),
        torch.randn(5, 16, 4, 1),
        torch.randn(5, 16, 4, 1),
    ]
    output = decoder(sample_data)
    print(f"output num: {len(output)}")
    print(f"output shape: {output[0].shape}")
