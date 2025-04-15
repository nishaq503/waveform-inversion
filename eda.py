#!/usr/bin/env -S uv run --script
#
# This script allows you perform some exploratory data analysis (EDA) on the
# data from the corresponding Kaggle competition. To run this script, you need
# to install `uv` and make this script executable.
#
# ## Usage
#
# ```bash
# chmod +x eda.py
# ./eda.py
# ```
#
# /// script
# requires-python = ">=3.10.12"
# dependencies = [
#     "waveform-inversion",
# ]
#
# [tool.uv.sources]
# waveform-inversion = { path = "." }
# ///


def main() -> None:
    print("Hello from eda.py!")


if __name__ == "__main__":
    main()
