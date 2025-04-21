"""Train a model using PyTorch."""

import json
import pathlib

import numpy
import torch

from tqdm.auto import tqdm

import waveform_inversion.utils as wi_utils

logger = wi_utils.make_logger(__name__)


def train(
    *,
    dl_train: torch.utils.data.DataLoader,
    dl_valid: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optim: torch.optim.Optimizer,
    n_epochs: int,
    device: torch.device,
    models_dir: pathlib.Path,
    final_model_path: pathlib.Path,
):
    """Train a model using PyTorch.

    Args:
        dl_train: The training data loader.
        dl_valid: The validation data loader.
        model: The model to train.
        criterion: The loss function.
        optim: The optimizer.
        n_epochs: The number of epochs to train for.
        device: The device to train on (CPU or GPU).
        models_dir: Directory to save the models.
        final_model_path: Path to save the final model.
    """
    logger.info("Starting training...")
    logger.info(f"Training on device: {device}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Final model path: {final_model_path}")

    model = model.to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "valid_loss": [],
    }
    for epoch in range(n_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{n_epochs}:")

        model.train()
        train_losses = []
        for inputs, targets in tqdm(dl_train, desc=f"train {epoch + 1}/{n_epochs}", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())

        train_loss: float = numpy.mean(train_losses)
        logger.info(f"Train loss: {train_loss:.3e}")

        model.eval()
        valid_losses = []
        for inputs, targets in tqdm(dl_valid, desc=f"valid {epoch + 1}/{n_epochs}", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.inference_mode():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            valid_losses.append(loss.item())

        valid_loss: float = numpy.mean(valid_losses)
        logger.info(f"Valid loss: {valid_loss:.3e}")

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} completed.")

        # Save the model
        model_path = models_dir / f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    logger.info("Training completed.")
    logger.info(f"Final training loss: {train_loss:.3e}")
    logger.info(f"Final validation loss: {valid_loss:.3e}")

    # Save the training history
    with (models_dir / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
