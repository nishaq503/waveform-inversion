"""Physics-guided ML models to solve full-waveform inversion problems."""

import logging
import pathlib

import typer

from typing_extensions import Annotated

import waveform_inversion as wi
import waveform_inversion.models as wi_models

wi.seed_all()

logger = wi.make_logger("waveform_inversion", level=wi.LOG_LEVEL)

# Add formatting to the logger
logging.basicConfig(
    filename=wi.LOG_FILE,
    level=wi.LOG_LEVEL,
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
    )] = wi.DATA_INP_DIR,
    work_dir: Annotated[pathlib.Path, typer.Option(
        "--work-dir",
        "-w",
        help="Working directory for the training data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    )] = wi.DATA_WORK_DIR,
    out_dir: Annotated[pathlib.Path, typer.Option(
        "--out-dir",
        "-o",
        help="Output directory for the training data.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    )] = wi.DATA_OUT_DIR,
    n_epochs: Annotated[int, typer.Option(
        "--n-epochs",
        "-e",
        help="Number of epochs to train the model.",
    )] = 50,
    retrain: Annotated[bool, typer.Option(
        "--retrain",
        "-r",
        help="Whether to retrain the model.",
    )] = False,
    eda: Annotated[bool, typer.Option(
        "--eda",
        "-d",
        help="Whether to perform exploratory data analysis.",
    )] = False,
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
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Retrain: {retrain}")

    models_dir = work_dir / "models"
    if not models_dir.exists():
        models_dir.mkdir()
        logger.info(f"Created new models directory: {models_dir}")
    else:
        logger.info(f"Using existing models directory: {models_dir}")

    # Print the environment variables loaded by dotenv
    logger.info(f"Loaded environment {len(wi.ENV_VARS)} variables:")
    for key, value in wi.ENV_VARS.items():
        logger.info(f"  {key}: {value}")

    if eda:
        pass
    else:
        wi_models.train_and_test(
            inp_dir=inp_dir,
            out_dir=out_dir,
            models_dir=models_dir,
            model=wi_models.DumbNet(),
            n_epochs=n_epochs,
            retrain=retrain,
            valid_frac=0.2,
        )


if __name__ == "__main__":
    # Run the CLI
    cli()
