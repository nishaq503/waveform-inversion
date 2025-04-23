"""Drawing some plots for EDA."""

import pathlib

import numpy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import waveform_inversion.utils as wi_utils

logger = wi_utils.make_logger(__name__)


def draw_io_pair(
    seismic_measurements: numpy.ndarray,
    velocity_map: numpy.ndarray,
    plots_dir: pathlib.Path,
    plot_name: str,
):
    """Draw pairs of input/output data for EDA.

    Args:
        seismic_measurements: Input data (seismic measurements).
        velocity_map: Output data (velocity map).
        plots_dir: Directory to save the plots.
        plot_name: Name of the plot file without extension.
    """
    logger.info(f"Drawing pair: {plot_name}")

    logger.info("Seismic data:")
    log_data_summary(seismic_measurements)

    logger.info("Velocity data:")
    log_data_summary(velocity_map)

    # The seismic measurements should have shape (5, 1000, 70), i.e., 5 channels
    # with 1000 time steps and 70 receivers.
    # The velocity map should have shape (70, 70) representing the ground truth
    # of the subsurface geology.
    # We will plot the seismic measurements and the velocity map side by side.

    fig: Figure
    axes: list[list[Axes]]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 10),
    )
    fig.suptitle("Seismic Measurements and Velocity Map")
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.875,
        bottom=0.05,
        hspace=0.25,
        wspace=0.25,
    )

    # Draw the seismic measurements
    for i in range(5):
        x = i % 3
        y = i // 3
        draw_subfigure(
            ax=axes[y][x],
            data=seismic_measurements[i],
            subtitle=f"Seismic Measurement {i + 1}",
            x_label="Receiver",
            y_label="Time step",
        )
    # Draw the velocity map
    draw_subfigure(
        ax=axes[1][2],
        data=velocity_map.squeeze(),
        subtitle="Velocity Map",
        x_label="X",
        y_label="Y",
    )

    # Save the plot
    plot_path = plots_dir / f"{plot_name}.png"
    fig.savefig(plot_path, dpi=300)
    logger.info(f"Saved plot {plot_path.name}")

    # Close the figure to free up memory
    plt.close(fig)


def draw_subfigure(
    *,
    ax: Axes,
    data: numpy.ndarray,
    subtitle: str,
    x_label: str,
    y_label: str,
):
    """Draw a subfigure for the given data.

    Args:
        ax: Axes object to draw the subfigure on.
        data: Data to plot.
        subtitle: Subtitle for the subfigure.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
    """

    ax.set_title(subtitle)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    im = ax.imshow(data, aspect="auto")
    plt.colorbar(im, ax=ax, orientation="vertical")

    # Show the x-ticks and label along the top edge instead of the bottom edge
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


def log_data_summary(data: numpy.ndarray):
    """Some logs about the data."""
    shape = data.shape
    dtype = data.dtype
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    std_val = data.std()
    median_val = numpy.median(data)

    logger.info(f"\tshape: {shape}, dtype: {dtype}, min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}, median: {median_val}")


__all__ = [
    "draw_io_pair",
]
