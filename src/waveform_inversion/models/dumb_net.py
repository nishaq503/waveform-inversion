"""The DumbNet model taken from https://www.kaggle.com/code/pshikk/dumbernet-sub
"""

from torch import nn
import torch

import waveform_inversion.utils as wi_utils

logger = wi_utils.make_logger(__name__)


class DumbNet(nn.Module):
    """An MLP model for the waveform inversion task.
    """
    def __init__(
            self,
            pool_size: tuple[int, int] = (10, 2),
            input_size: int = 5 * 1000 * 70,
            hidden_sizes: list[int] = [
                256 * 70,
                196 * 70,
                128 * 70,
                98 * 70,
            ],
            output_size: int = 70 * 70,
        ):
        """Initialize the DumbNet model.

        Args:
            pool_size: Size of the pooling layer.
            input_size: Size of the input tensor.
            hidden_size: Size of the hidden layer.
            output_size: Size of the output layer.
        """
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        in_size = input_size // (pool_size[0] * pool_size[1])
        dropout_rate = 0.5

        layers = [nn.Linear(in_size, hidden_sizes[0])]
        for hs_in, hs_out in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hs_in, hs_out))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the DumbNet model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the model.
        """
        bs = x.shape[0]

        # We apply a pool to reduze input size
        x_pool = self.pool(x)

        #Model is just a
        out = self.model(x_pool.view(bs, -1))

        return out.view(bs, 1, 70, 70) * 1000 + 1500


__all__ = ["DumbNet"]
