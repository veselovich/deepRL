"""
Utility functions to make predictions.
"""

import requests
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_source: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = torch.device("cpu"),
):
    """Predicts on a target image (from a file path or URL) with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_source (str): Filepath or URL of the target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms.Compose, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to "cpu".

    Returns:
        dict: Information about the image and prediction.
    """
    # Load image
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_source)

    # Apply transformation
    image_transform = transform or transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Prepare model
    model.to(device)
    model.eval()

    # Transform image and add batch dimension
    transformed_image = image_transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.inference_mode():
        logits = model(transformed_image)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    # Plot prediction
    predicted_label = class_names[pred_label]
    predicted_probability = probs.max().item()

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {predicted_label} | Prob: {predicted_probability:.3f}")
    plt.axis(False)
    plt.show()

    return {
        "predicted_label": predicted_label,
        "predicted_probability": predicted_probability,
        "image_shape": img.size,
        "transformed_image": transformed_image,
    }


def plot_confusion_matrix_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "Blues",
) -> None:
    """Computes and plots the confusion matrix for a PyTorch model using mlxtend.

    Args:
        model: A PyTorch model to be evaluated.
        dataloader: A DataLoader instance for the test dataset.
        num_classes: The number of classes in the dataset.
        device: A target device to compute on (e.g., "cuda" or "cpu").
        class_names: Optional list of class names for the confusion matrix labels.
        figsize: Tuple specifying the size of the confusion matrix plot.
        cmap: Colormap for the confusion matrix visualization.
    """
    model.to(device)
    model.eval()

    confusion_matrix_metric = ConfusionMatrix(
        num_classes=num_classes, task="multiclass"
    ).to(device)

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            confusion_matrix_metric.update(preds, y)

    conf_matrix = confusion_matrix_metric.compute().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    plot_confusion_matrix(
        conf_mat=conf_matrix, class_names=class_names, cmap=cmap, axis=ax
    )
    plt.title("Confusion Matrix")
    plt.show()


def top_k_fails(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device = torch.device("cpu"),
    k: int = 5,
) -> dict:
    """Identifies and displays the top k images recognized incorrectly with the highest probabilities.

    Args:
        model (torch.nn.Module): A trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to run the model on (e.g., "cuda" or "cpu").
        class_names (List[str]): List of class names corresponding to the labels.
        k (int): Number of top incorrect predictions to analyze and display.

    Returns:
        dict: Contains incorrectly predicted images, probabilities, labels, and predictions.
    """
    model.to(device)
    model.eval()

    incorrect_images, incorrect_probs, incorrect_labels, incorrect_preds = (
        [],
        [],
        [],
        [],
    )

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            incorrect_mask = preds != y
            if incorrect_mask.any():
                incorrect_images.extend(X[incorrect_mask].cpu())
                incorrect_probs.extend(
                    probs[incorrect_mask, preds[incorrect_mask]].cpu()
                )
                incorrect_labels.extend(y[incorrect_mask].cpu())
                incorrect_preds.extend(preds[incorrect_mask].cpu())

    incorrect_probs = torch.tensor(incorrect_probs)
    sorted_indices = torch.argsort(incorrect_probs, descending=True)[:k]

    images = torch.stack([incorrect_images[i] for i in sorted_indices])
    probs = incorrect_probs[sorted_indices]
    labels = torch.tensor([incorrect_labels[i] for i in sorted_indices])
    predictions = torch.tensor([incorrect_preds[i] for i in sorted_indices])

    plt.figure(figsize=(15, 10))
    for i in range(min(k, len(images))):
        image = images[i]
        prob, label, pred = probs[i].item(), labels[i].item(), predictions[i].item()

        plt.subplot(1, k, i + 1)
        plt.imshow(torch.clip(image.permute(1, 2, 0), 0, 1))
        plt.title(
            f"Pred: {class_names[pred]} ({prob:.2f})\nLabel: {class_names[label]}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "images": images,
        "probs": probs,
        "labels": labels,
        "predictions": predictions,
    }
