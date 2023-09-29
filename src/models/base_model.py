import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()

        #################################################################################
        # TODO: Q1.b) Print the number of trainable parameters for each layer and total number of trainable parameters
        # Simply update the ret_str by adding new lines to it.
        #################################################################################
        ret_str = ""

        total_params = 0
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                num_params = parameter.numel()
                total_params += num_params
                ret_str = ret_str + f"Layer: {name} | Trainable Parameters: {num_params}" +'\n'


        ret_str = ret_str + f"Total Trainable Parameters: {total_params}"
        
        return ret_str