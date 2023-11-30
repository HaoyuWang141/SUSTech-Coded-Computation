import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Base class for implementing decoders.
    """

    def __init__(self, num_in, num_out):
        """
        Parameters
        ----------
            num_in: int
                Number of input units for a forward pass of the coder.
            num_out: int
                Number of output units from a forward pass of the coder.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, datasets):
        """
        Parameters
        ----------
            datasets: ``torch.autograd.Variable``
                Input data for a forward pass of the coder.
        """
        pass
