# Geophysical Waveform Inversion

## Pre-requisites

- [`uv`](https://docs.astral.sh/uv/getting-started/)
- `Python 3.10.12`: `uv` should handle this.

## Usage

Clone the repository, `cd` into it, and install the dependencies using `uv`:

```bash
uv sync
```

Set up the environment variables:

```bash
uv run dotenv set DATA_INP_DIR path/to/data/input/waveform-inversion
uv run dotenv set DATA_WORK_DIR path/to/data/working/waveform-inversion
uv run dotenv set LOG_LEVEL DEBUG
```

- The `DATA_INP_DIR` should contain the path to the input data directory, which was downloaded from kaggle.
- The `DATA_WORK_DIR` should be a directory where the package can store intermediate files and results.
- The `LOG_LEVEL` can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.

The `DATA_INP_DIR` and `DATA_WORK_DIR` can be absolute paths or relative to the current working directory.
If they are relative, they will be resolved to absolute paths when any script is first run.

## Running scripts

### The example script

```bash
uv run example
```
