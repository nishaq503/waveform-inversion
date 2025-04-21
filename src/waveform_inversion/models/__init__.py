"""ML models for performing full-waveform inversion (FWI) problems."""

from .dumb_net import DumbNet
from .train import train


__all__ = [
    "DumbNet",
    "train",
]
