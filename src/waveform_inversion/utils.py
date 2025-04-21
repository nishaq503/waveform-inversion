"""Helper functions for the package."""

import logging
import os
import pathlib

import dotenv


def handle_dotenv(path: pathlib.Path) -> dict[str, str]:
    """Handle the .env file for the package."""

    if not path.exists():
        msg = f"File {path} not found. Please create a .env file in the root directory."
        raise FileNotFoundError(msg)
    dotenv.load_dotenv(path)

    dotenv_vars = dotenv.dotenv_values(path)

    if not (dotenv_vars and len(dotenv_vars) > 0):
        raise ValueError("No environment variables loaded. Please check if the .env file exists.")

    # Some variables' names start with "DATA" and their values are relative to ~/Documents/kaggle. Convert them to absolute paths.
    for key, value in dotenv_vars.items():
        if key.startswith("DATA") and not value.startswith("/"):
            path = (pathlib.Path.home() / "Documents" / "kaggle" / value).resolve()
            if not path.exists():
                msg = f"File {path} not found. Please check the .env file."
                raise FileNotFoundError(msg)
            dotenv_vars[key] = str(path)

    # Write the updated variables to the .env file
    with open(".env", "w") as f:
        for key, value in dotenv_vars.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = value

    return dotenv_vars


ENV_VARS = handle_dotenv((pathlib.Path(__file__).parents[2] / ".env").resolve())
DATA_INP_DIR = pathlib.Path(ENV_VARS["DATA_INP_DIR"])
DATA_WORK_DIR = pathlib.Path(ENV_VARS["DATA_WORK_DIR"])
DATA_OUT_DIR = pathlib.Path(ENV_VARS["DATA_OUT_DIR"])
LOG_LEVEL = ENV_VARS["LOG_LEVEL"]


def make_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """Create a logger for the package.

    Args:
        name: Name of the logger.
        level: Logging level.

    Returns:
        Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def seed_all(seed: int = 42) -> None:
    """Set the random seed for all libraries.

    Args:
        seed: Random seed.
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "handle_dotenv",
    "ENV_VARS",
    "DATA_INP_DIR",
    "DATA_WORK_DIR",
    "LOG_LEVEL",
    "make_logger",
    "seed_all",
]
