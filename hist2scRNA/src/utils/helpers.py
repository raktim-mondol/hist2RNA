"""
Helper functions for hist2scRNA

General utility functions for various tasks.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str: str = 'cuda'):
    """
    Get torch device

    Args:
        device_str: 'cuda' or 'cpu'

    Returns:
        torch.device
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def create_output_dirs(output_dir: str, checkpoint_dir: str = None):
    """
    Create output directories

    Args:
        output_dir: main output directory
        checkpoint_dir: checkpoint directory (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Created output directories at {output_dir}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format

    Args:
        seconds: time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
