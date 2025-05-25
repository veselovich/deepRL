"""
Contains functions for training and testing a PyTorch model.
"""

import os
import time
import torch
import pandas as pd

from datetime import datetime
from tqdm.auto import tqdm
from typing import Dict, Optional
from torchmetrics import F1Score, Precision, Recall
from torch.utils.tensorboard.writer import SummaryWriter


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training loss and training accuracy metrics.
    For example:

    {'train_loss': 0.0223, 'train_acc': 0.8985}
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return {"train_loss": train_loss, "train_acc": train_acc}


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    precision_metric: Optional[Precision] = None,
    recall_metric: Optional[Recall] = None,
    f1_metric: Optional[F1Score] = None,
) -> Dict[str, float]:
    """Tests a PyTorch model for a single epoch with optional additional metrics.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    precision_metric: An optional Precision metric from torchmetrics.
    recall_metric: An optional Recall metric from torchmetrics.
    f1_metric: An optional F1Score metric from torchmetrics.

    Returns:
    A dictionary of testing metrics including loss, accuracy, and any provided metrics.
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Reset metrics if provided
    if precision_metric:
        precision_metric.reset()
    if recall_metric:
        recall_metric.reset()
    if f1_metric:
        f1_metric.reset()

    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate predictions and accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # 4. Update additional metrics if provided
            if precision_metric:
                precision_metric.update(test_pred_labels, y)
            if recall_metric:
                recall_metric.update(test_pred_labels, y)
            if f1_metric:
                f1_metric.update(test_pred_labels, y)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # Create results dictionary
    results = {"test_loss": test_loss, "test_acc": test_acc}

    # Add optional metrics to the results
    if precision_metric:
        results["precision"] = precision_metric.compute().item()
    if recall_metric:
        results["recall"] = recall_metric.compute().item()
    if f1_metric:
        results["f1_score"] = f1_metric.compute().item()

    return results


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
) -> pd.DataFrame:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints, and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g., "cuda" or "cpu").
    writer: (Optional) A TensorBoard SummaryWriter instance for logging metrics.

    Returns:
    A pandas DataFrame containing training and testing metrics for each epoch.
    """
    # Create empty results list
    results = []

    # Make sure the model is on the target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        metrics = {"epoch": epoch + 1}

        # Perform a training step
        train_result = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        metrics.update(train_result)

        # Perform a testing step
        test_result = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        metrics.update(test_result)

        # Compute epoch time
        epoch_time = time.time() - start_time
        metrics["epoch_time"] = epoch_time
        results.append(metrics)

        # Print results
        print(
            " | ".join(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in metrics.items()
            )
        )

        # Log metrics to TensorBoard
        if writer:
            try:
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={
                        "train_loss": train_result["train_loss"],
                        "test_loss": test_result["test_loss"],
                    },
                    global_step=epoch,
                )
                writer.add_scalars(
                    main_tag="Accuracy",
                    tag_scalar_dict={
                        "train_acc": train_result["train_acc"],
                        "test_acc": test_result["test_acc"],
                    },
                    global_step=epoch,
                )
                writer.add_scalar("Epoch Time", epoch_time, global_step=epoch)
            except KeyError as e:
                print(f"KeyError logging to TensorBoard: {e}")

    # Close the writer if provided
    if writer:
        writer.close()

    # Return the filled results as a pandas DataFrame
    return pd.DataFrame(results)


def create_writer(data_name: str, model_name: str, extra: str = None) -> SummaryWriter:
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter instance with a specific log_dir structure.

    Args:
        data_name (str): Name of the dataset.
        model_name (str): Name of the model.
        extra (str, optional): Additional information for the directory structure. Defaults to None.

    Returns:
        SummaryWriter: Writer instance saving logs to the specified directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")  # Current date in YYYY-MM-DD format
    log_dir_components = ["runs", timestamp, data_name, model_name]
    if extra:
        log_dir_components.append(extra)
    log_dir = os.path.join(*log_dir_components)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
