import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_conv_segment(self) -> nn.Sequential:
        '''
        Get the conv segment of the model.
        '''
        raise NotImplementedError
    
    def get_fc_segment(self) -> nn.Sequential:
        '''
        Get the fc segment of the model.
        '''
        raise NotImplementedError
    
    def calculate_conv_output(self, input_dim: tuple[int]) -> tuple[int, int, int]:
        '''
        Calculate the output shape of the conv segment.
        
        :param input_dim: the input shape of the conv segment, (channel, height, width)
        
        :return: the output shape of the conv segment, (channel, height, width)
        '''
        raise NotImplementedError