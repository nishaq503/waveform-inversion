"""ML models for performing full-waveform inversion (FWI) problems."""

import csv
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import waveform_inversion.data as wi_data
import waveform_inversion.utils as wi_utils
from .dumb_net import DumbNet
from .train import train

logger = wi_utils.make_logger(__name__)


def train_and_test(
    *,
    inp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    models_dir: pathlib.Path,
    model: nn.Module,
    n_epochs: int,
    retrain: bool,
    valid_frac: float,
) -> nn.Module:
    """Train and test the model, and save a submission file.

    Args:
        inp_dir: Input directory for the training data.
        out_dir: Output directory for the training data.
        models_dir: Directory to save the trained model.
        model: Model to be trained and tested.
        n_epochs: Number of epochs for training.
        retrain: Whether to retrain the model if it already exists.

    Returns:
        model: The trained model.
    """
    io_paths = wi_data.get_train_paths(inp_dir)
    logger.info(f"Found {len(io_paths)} input/output paths.")

    train_paths, valid_paths = wi_data.train_valid_split(io_paths, valid_frac)
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

        train(
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
    return model


__all__ = [
    "DumbNet",
    "train",
    "train_and_test",
]
