"""Some utility functions for the dataset."""

import pathlib
import random
import typing

from waveform_inversion.utils import LOG_LEVEL, make_logger

logger = make_logger(__name__, level=LOG_LEVEL)

T = typing.TypeVar("T")


def get_train_paths(inp_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Get the paths to the training data files."""
    path_pairs = [
        (
            p,
            pathlib.Path(str(p).replace('seis', 'vel').replace('data', 'model'))
        )
        for p in
        (inp_dir / "train_samples").rglob("*.npy")
        if ("seis" in p.stem) or ("data" in p.stem)
    ]

    for inp_path, out_path in path_pairs:
        if not inp_path.exists():
            msg = f"Input file {inp_path} not found."
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not out_path.exists():
            msg = f"Output file {out_path} not found."
            logger.error(msg)
            raise FileNotFoundError(msg)

    logger.debug(f"Found {len(path_pairs)} pairs of input and output files.")
    return path_pairs


def train_valid_split(values: list[T], valid_frac: float) -> tuple[list[T], list[T]]:
    """Split a list into training and validation sets."""
    if not (0 < valid_frac < 1):
        raise ValueError("`valid_frac` must be between 0 and 1")

    # Shuffle the values
    random.shuffle(values)

    # Calculate the split index
    split_idx = int(len(values) * (1 - valid_frac))

    # Split the values
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    logger.debug(f"Split {len(values)} items into {len(train_values)} training and {len(test_values)} validation items.")
    return train_values, test_values


__all__ = [
    "get_train_paths",
    "train_valid_split",
]
