"""Physics-guided ML models to solve full-waveform inversion problems."""

import csv
import logging
import pathlib

import torch
import typer

from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing_extensions import Annotated

import waveform_inversion as wi
import waveform_inversion.data as wi_data
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

    io_paths = wi.data.get_train_paths()
    logger.info(f"Found {len(io_paths)} input/output paths.")

    train_paths, valid_paths = wi.data.train_test_split(io_paths, test_frac=0.2)
    logger.info(f"Split {len(io_paths)} paths into {len(train_paths)} for training and {len(valid_paths)} for validation.")

    logger.debug("Training paths:")
    for inp_path, out_path in train_paths:
        logger.debug(f"Input path: {inp_path}")
        logger.debug(f"Output path: {out_path}")

    logger.debug("Validation paths:")
    for inp_path, out_path in valid_paths:
        logger.debug(f"Input path: {inp_path}")
        logger.debug(f"Output path: {out_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    final_model_path = out_dir / "final_model.pth"
    model = wi_models.DumbNet()
    logger.info(f"Final model path: {final_model_path}")

    if not final_model_path.exists() or retrain:
        dl_train = DataLoader(
            wi_data.SeismicDataset(train_paths),
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=10,
            persistent_workers=True,
        )
        dl_valid = DataLoader(
            wi_data.SeismicDataset(valid_paths),
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=10,
            persistent_workers=True,
        )

        criterion = nn.L1Loss()
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

        wi_models.train(
            dl_train=dl_train,
            dl_valid=dl_valid,
            model=model,
            criterion=criterion,
            optim=optim,
            n_epochs=n_epochs,
            device=device,
            models_dir=models_dir,
            final_model_path=final_model_path,
        )
        model.eval()
    else:
        logger.info(f"Final model already exists at {final_model_path}. Skipping training.")
        model.eval()
        model.load_state_dict(torch.load(final_model_path, weights_only=True))

    test_files = list((inp_dir / "test").glob("*.npy"))
    dl_test = DataLoader(
        wi_data.TestDataset(test_files),
        batch_size=8,
        num_workers=10,
        pin_memory=True,
    )
    with (out_dir / "submission.csv").open("wt", newline="") as f:
        x_cols = [f'x_{i}' for i in range(1, 70, 2)]
        fieldnames = ['oid_ypos'] + x_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for inputs, ids in tqdm(dl_test, desc="test"):
            inputs = inputs.to(device)
            with torch.inference_mode():
                outputs = model(inputs)
                y_preds = outputs[:, 0].cpu().numpy()

            for y_pred, id in zip(y_preds, ids):
                for y_pos in range(70):
                    row = dict(
                        zip(
                            x_cols,
                            [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                        )
                    )
                    row['oid_ypos'] = f"{id}_y_{y_pos}"

                    writer.writerow(row)

    logger.info("Submission file created.")
    return


if __name__ == "__main__":
    # Run the CLI
    cli()
