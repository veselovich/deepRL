import torch
import torch.nn as nn
import torchvision
from torch import nn
from torchinfo import summary
from typing import Type, Optional, Tuple
from pathlib import Path
from src.utils import print_in_columns


def pretrained_is_available(model_name: str) -> bool:
    """
    Checks if pretrained weights are available for a specified model.

    Args:
        model_name (str): Name of the model to check.

    Returns:
        bool: True if pretrained weights are available; raises ValueError otherwise.
    """
    models_with_weights = [
        model
        for model in torchvision.models.list_models()
        if len(torchvision.models.get_model_weights(model)) > 0
    ]

    if model_name in models_with_weights:
        return True
    else:
        print("Available models:")
        print_in_columns(models_with_weights, columns=3)
        raise ValueError(
            f"Model '{model_name}' not found or does not have pretrained weights in torchvision.models.\n"
            "Please choose a model from the list above."
        )


def get_vision_weights(model_name: str):
    """
    Retrieves the default pretrained weights for a specified model.

    Args:
        model_name (str): Name of the model.

    Returns:
        Any: Pretrained weights object.
    """
    if pretrained_is_available(model_name):
        return torchvision.models.get_model_weights(model_name)["DEFAULT"]


def create_vision_model(
    model_name: str,
    out_features: int,
    device: torch.device,
    weights=None,
    seed: Optional[int] = None,
    print_summary: bool = False,
    compile: bool = False,
) -> Tuple[nn.Module, torchvision.transforms.Compose]:
    """
    Creates a vision model feature extractor with a customizable classifier head.

    Args:
        model_name (str): Name of the pretrained model (e.g., 'efficientnet_b0', 'resnet50').
        out_features (int): Number of output features for the classifier head.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        seed (Optional[int]): Seed for reproducibility.
        print_summary (bool): Whether to print the model summary.
        compile (bool): Whether to compile the model using Torch 2.0.

    Returns:
        model (torch.nn.Module): The created vision model.
        transform (torchvision.transforms.Compose): Preprocessing transforms for the model.
    """
    if weights is None:
        weights = get_vision_weights(model_name)

    # Load the model with pretrained weights
    model_constructor = getattr(torchvision.models, model_name, None)
    if not model_constructor:
        raise ValueError(
            f"'{model_name}' is not a valid model name in torchvision.models."
        )

    model = model_constructor(weights=weights).to(device)

    # Extract preprocessing transforms
    transform = weights.transforms()

    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

    # Modify the classifier head based on the model architecture
    if hasattr(model, "fc"):  # ResNet-like architectures
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features).to(device)
    elif hasattr(model, "classifier"):  # EfficientNet, MobileNet, etc.
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features, out_features)
        ).to(device)
    elif hasattr(model, "heads"):  # Vision Transformer, etc.
        in_features = model.heads[-1].in_features
        model.heads = nn.Linear(in_features, out_features).to(device)
    else:
        raise ValueError(f"Unsupported model architecture for '{model_name}'.")

    model.name = model_name
    print(f"[INFO] Created new {model.name} model.")

    if print_summary:
        summary(
            model,
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )

    if compile:
        try:
            model = torch.compile(model)
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            model.eval()
            with torch.no_grad():
                model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[INFO] Model {model.name} is compiled.")
        except Exception as e:
            print(f"[WARNING] Error compiling model: {e}")

    return model, transform


def load_model(
    model_class: Type[nn.Module], model_path: str, device: torch.device = torch.device("cpu")
) -> Optional[nn.Module]:
    """
    Loads a PyTorch model from a file.

    Args:
        model_class (Type[nn.Module]): The class of the model to instantiate.
        model_path (str): The path to the saved model file.
        device (str): The device to map the model to ('cpu' or 'cuda').

    Returns:
        Optional[nn.Module]: The loaded PyTorch model, or None if loading fails.
    """
    model = model_class()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model


def save_model(model: nn.Module, target_dir: str, model_name: str) -> str:
    """
    Saves a PyTorch model to a target directory and returns the absolute path to the saved model.

    Args:
        model (nn.Module): A target PyTorch model to save.
        target_dir (str): Directory for saving the model.
        model_name (str): Filename for the saved model (must end with ".pth" or ".pt").

    Returns:
        str: Absolute path to the saved model file.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    return str(model_save_path.resolve())
