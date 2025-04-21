"""Physics-guided ML models to solve full-waveform inversion problems."""

import logging

from . import utils
from . import data

utils.seed_all()
logger = utils.make_logger("waveform_inversion", level=utils.ENV_VARS["LOG_LEVEL"])


def main():
    """List all environment variables loaded by dotenv."""

    # Add formatting to the logger
    logging.basicConfig(
        level=utils.ENV_VARS["LOG_LEVEL"],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.debug("Debugging information.")
    logger.info("Logger initialized.")
    logger.warning("Warning message.")
    logger.error("Error message.")
    logger.critical("Critical message.")

    # Print the environment variables loaded by dotenv
    logger.info(f"Loaded environment {len(utils.ENV_VARS)} variables:")
    for key, value in utils.ENV_VARS.items():
        logger.info(f"  {key}: {value}")

    io_paths = data.get_train_paths()
    logger.info(f"Found {len(io_paths)} input/output paths.")
    for inp_path, out_path in io_paths:
        logger.debug(f"Input path: {inp_path}")
        logger.debug(f"Output path: {out_path}")
