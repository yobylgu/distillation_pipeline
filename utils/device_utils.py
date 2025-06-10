"""
Device setup utilities for the distillation pipeline.
"""
import os
import torch

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_device():
    """Setup computing device."""
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
