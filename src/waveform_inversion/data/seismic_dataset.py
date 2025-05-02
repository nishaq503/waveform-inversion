"""The main dataset class for the seismic waveform inversion task."""

import pathlib

import numpy
from torch.utils.data import Dataset

import waveform_inversion.utils as wi_utils

logger = wi_utils.make_logger(__name__)


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
        """Return the total number of examples in the dataset."""
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Get the seismic data and corresponding label.

        Args:
            idx: The index of the data point to retrieve.

        Returns:
            A tuple containing the seismic data and the corresponding label.
        """
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        train_x: numpy.ndarray = numpy.load(self.inputs_files[file_idx], mmap_mode='r')
        train_y: numpy.ndarray = numpy.load(self.output_files[file_idx], mmap_mode='r')

        try:
            x, y = train_x[sample_idx].copy(), train_y[sample_idx].copy()
            # Take only every other time-step from the x
            # x = x[:, ::2, :]
            return x, y
        finally:
            del train_x, train_y


__all__ = [
    "SeismicDataset",
]
