"""Physics-guided ML models to solve full-waveform inversion problems."""

import logging
import pathlib

import typer
from typing_extensions import Annotated

import waveform_inversion as wi

wi.utils.seed_all()

logger = wi.utils.make_logger("waveform_inversion", level=wi.utils.LOG_LEVEL)

# Add formatting to the logger
logging.basicConfig(
    level=wi.utils.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

cli = typer.Typer(
    name="waveform-inversion",
    help="Physics-guided ML models to solve full-waveform inversion problems.",
)


@cli.command()
def main(
    inp_dir: Annotated[pathlib.Path, typer.Option(
        "--inp-dir",
        "-i",
        help="Input directory for the training data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    )] = wi.utils.DATA_INP_DIR,
    work_dir: Annotated[pathlib.Path, typer.Option(
        "--work-dir",
        "-w",
        help="Working directory for the training data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    )] = wi.utils.DATA_WORK_DIR,
    out_dir: Annotated[pathlib.Path, typer.Option(
        "--out-dir",
        "-o",
        help="Output directory for the training data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    )] = wi.utils.DATA_OUT_DIR,
):
    """List all environment variables loaded by dotenv."""

    logger.debug("Debugging information.")
    logger.info("Logger initialized.")
    logger.warning("Warning message.")
    logger.error("Error message.")
    logger.critical("Critical message.")

    logger.info("Starting the main function.")
    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Output directory: {out_dir}")

    # Print the environment variables loaded by dotenv
    logger.info(f"Loaded environment {len(wi.utils.ENV_VARS)} variables:")
    for key, value in wi.utils.ENV_VARS.items():
        logger.info(f"  {key}: {value}")

    io_paths = wi.data.get_train_paths()
    logger.info(f"Found {len(io_paths)} input/output paths.")
    for inp_path, out_path in io_paths:
        logger.debug(f"Input path: {inp_path}")
        logger.debug(f"Output path: {out_path}")


if __name__ == "__main__":
    # Run the CLI
    cli()
