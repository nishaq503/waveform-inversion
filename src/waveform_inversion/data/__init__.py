"""Handling the data for the Kaggle competition."""

import pathlib
import random

from torch.utils.data import Dataset

import numpy
from waveform_inversion import utils

logger = utils.make_logger(__name__, level=utils.ENV_VARS["LOG_LEVEL"])


def get_train_paths() -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Get the paths to the training data files."""
    path_pairs = [
        (
            p,
            pathlib.Path(str(p).replace('seis', 'vel').replace('data', 'model'))
        )
        for p in
        (utils.DATA_INP_DIR / "train_samples").rglob("*.npy")
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


def train_test_split(values: list, test_frac: float) -> tuple[list, list]:
    """Split a list into a training and test set."""
    if not (0 < test_frac < 1):
        raise ValueError("`test_frac` must be between 0 and 1")

    # Shuffle the values
    random.shuffle(values)

    # Calculate the split index
    split_idx = int(len(values) * (1 - test_frac))

    # Split the values
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    logger.debug(f"Split {len(values)} values into {len(train_values)} training and {len(test_values)} test values.")
    return train_values, test_values


class SeismicDataset(Dataset):
    """A dataset for the seismic waveform inversion task."""
    def __init__(self, io_paths: list[tuple[pathlib.Path, pathlib.Path]], n_examples_per_file: int = 500):
        """Initialize the dataset.

        Args:
            io_paths: A list of tuples containing the input and output file paths.
            n_examples_per_file: The number of examples per file. Default is 500, as per the Kaggle competition.
        """

        self.inputs_files, self.output_files = zip(*io_paths)
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        train_x: numpy.ndarray = numpy.load(self.inputs_files[file_idx], mmap_mode='r')
        train_y: numpy.ndarray = numpy.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return train_x[sample_idx].copy(), train_y[sample_idx].copy()
        finally:
            del train_x, train_y


__all__ = [
    "get_train_paths",
    "train_test_split",
    "SeismicDataset",
]
