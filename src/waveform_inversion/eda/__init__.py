"""Exploratory Data Analysis (EDA) for the Waveform Inversion competition."""

import pathlib

import waveform_inversion.data as wi_data
import waveform_inversion.utils as wi_utils
from . import plots

logger = wi_utils.make_logger(__name__)


def explore(
    *,
    inp_dir: pathlib.Path,
    work_dir: pathlib.Path,
):
    """Run exploratory data analysis.

    Args:
        inp_dir: Input directory for the training data.
        work_dir: Working directory for the EDA.
    """
    eda_dir = work_dir / "eda"
    eda_dir.mkdir(exist_ok=True)
    logger.info(f"Using {eda_dir} for EDA.")

    io_paths = wi_data.get_train_paths(inp_dir)
    logger.info(f"Found {len(io_paths)} input/output paths.")

    logger.debug("Input/output paths:")
    for inp_path, out_path in io_paths:
        logger.debug(f"Input path: {inp_path}")
        logger.debug(f"Output path: {out_path}")

    dataset = wi_data.SeismicDataset(io_paths)
    logger.info(f"Dataset size: {len(dataset)}")

    for i, (inp_data, out_data) in enumerate(dataset):
        logger.info(f"Processing input/output pair {i + 1}/{len(dataset)}")
        plots.draw_io_pair(inp_data, out_data, eda_dir, f"pair_{i + 1}.png")

        if i >= 100:
            logger.info("Processed enough pairs, stopping EDA.")
            break


__all__ = [
    "explore",
]
