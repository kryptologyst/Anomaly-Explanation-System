"""Device management and seeding utilities for reproducible experiments."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If requested device is not available
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS is not available")
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_reproducibility(config: DictConfig) -> torch.device:
    """
    Setup reproducibility and device configuration.
    
    Args:
        config: Configuration object containing seed and device settings
        
    Returns:
        torch.device: The configured device
    """
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    return device


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Device information including availability and properties
    """
    info = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["cuda"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
    
    return info
