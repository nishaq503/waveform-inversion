"""The test dataset class for the seismic waveform inversion task."""

import pathlib

import numpy
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """A dataset for the test data in the Kaggle competition."""
    def __init__(self, test_files: list[pathlib.Path]):
        """Initialize the test dataset.

        Args:
            test_files: A list of test file paths.
        """
        self.test_files = test_files


    def __len__(self):
        """Return the number of test files."""
        return len(self.test_files)


    def __getitem__(self, i: int) -> tuple[numpy.ndarray, str]:
        """Get the test data and file name.

        Args:
            i: The index of the test file.

        Returns:
            A tuple containing the test data and the file name.
        """
        test_file = self.test_files[i]

        return numpy.load(test_file), test_file.stem


__all__ = [
    "TestDataset",
]
