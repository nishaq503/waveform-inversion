"""Physics-guided ML models to solve full-waveform inversion problems."""

import pathlib
import dotenv

dotenv.load_dotenv()


def hello():
    """List all environment variables loaded by dotenv."""
    dotenv_vars = handle_dotenv()

    print(f"Loaded environment {len(dotenv_vars)} variables:")
    for key, value in dotenv_vars.items():
        print(f"  {key}: {value}")


def handle_dotenv() -> dict[str, str]:
    """Handle the .env file for the package."""
    dotenv_vars = dotenv.dotenv_values()

    if not (dotenv_vars and len(dotenv_vars) > 0):
        raise ValueError(
            "No environment variables loaded. Please check if the .env file exists."
        )

    # Some variables' names start with "DATA" and their values are relative to ~/Documents/kaggle. Convert them to absolute paths.
    for key, value in dotenv_vars.items():
        if key.startswith("DATA") and not value.startswith("/"):
            path = pathlib.Path.home() / "Documents" / "kaggle" / value
            dotenv_vars[key] = str(path.resolve())

    # Write the updated variables to the .env file
    with open(".env", "w") as f:
        for key, value in dotenv_vars.items():
            f.write(f"{key}={value}\n")

    return dotenv_vars
