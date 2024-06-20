import torch
from torch import nn

from decoder.decoder import Decoder


class MLPDecoder(Decoder):
    def __init__(self, num_in: int, num_out: int, in_dim: tuple) -> None:
        super().__init__(num_in, num_out, in_dim)
        self.nn = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU(),
            nn.Linear(num_out, num_out),
            nn.ReLU(),
            nn.Linear(num_out, num_out),
        )

    def forward(self, datasets: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(datasets) == 0 or self.num_out == 0:
            return []
        datasets = torch.stack(datasets, dim = -1)
        original_batch_size = datasets.size(0)
        batch_size = datasets.size(0) * datasets.size(1) * datasets.size(2) * datasets.size(3)
        datasets = datasets.view(batch_size, -1)
        out = self.nn(datasets)
        out = out.view(original_batch_size, *self.out_dim, self.num_out)
        out = list(torch.unbind(out, dim = -1))
        return out


if __name__ == "__main__":
    decoder = MLPDecoder(6, 4, (16, 4, 1))
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
