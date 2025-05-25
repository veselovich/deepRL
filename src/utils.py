import os
import torch


def get_device():
    """
    Determine the appropriate device for PyTorch operations.

    Returns:
        torch.device: The best available device ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculates the size of a model's state dictionary in megabytes (MB).

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        float: Size of the model in MB.
    """
    tmp_path = "tmp_model.pth"
    torch.save(model.state_dict(), tmp_path)
    model_size = os.path.getsize(tmp_path) / (1024 * 1024)  # Convert bytes to MB
    os.remove(tmp_path)
    return model_size


# Function to print list in multiple columns
def print_in_columns(items, columns=3):
    max_length = max(len(item) for item in items) + 2  # Padding for readability
    rows = (len(items) + columns - 1) // columns
    for row in range(rows):
        line = ""
        for col in range(columns):
            idx = row + col * rows
            if idx < len(items):
                line += items[idx].ljust(max_length)
        print(line)
