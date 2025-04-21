"""The DumbNet model taken from https://www.kaggle.com/code/pshikk/dumbernet-sub
"""

from torch import nn
import torch

import waveform_inversion.utils as wi_utils

logger = wi_utils.make_logger(__name__)


class ResidualBlock(nn.Module):
    """A residual block with optional downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        downsample: Whether to downsample the input. Default is False.
    """
    def __init__(self, in_channels: int, out_channels: int, device: torch.device, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device),
                nn.BatchNorm2d(out_channels, device=device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the residual block.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class DumbNet(nn.Module):
    """A deep CNN with residual blocks and a dense classifier.
    """
    def __init__(
            self,
            pool_size: tuple[int, int] = (8, 2),
            input_size: int = 5 * 1000 * 70,
            hs1: int = 540 * 70,
            hs2: int = 280 * 70,
            hs3: int = 140 * 70,
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

        self.model = nn.Sequential(
            nn.Linear(input_size // (pool_size[0] * pool_size[1]), hs1),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hs1, hs2),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hs2, hs3),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hs3, output_size),
        )

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
