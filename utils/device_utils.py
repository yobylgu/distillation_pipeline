"""
Device setup utilities for the distillation pipeline.
"""
import os
import torch

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_device(device_config='auto'):
    """
    Setup computing device with cluster and CUDA support.
    
    Args:
        device_config (str): Device configuration ('auto', 'cpu', 'cuda', 'mps', or specific GPU like 'cuda:0')
    
    Returns:
        torch.device: Configured device
    """
    # Handle CUDA_VISIBLE_DEVICES for cluster environments
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible is not None:
        print(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible}")
    
    # Handle SLURM environment variables
    slurm_local_id = os.environ.get('SLURM_LOCALID')
    if slurm_local_id is not None:
        print(f"Running in SLURM environment, local ID: {slurm_local_id}")
    
    if device_config == 'auto':
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    elif device_config.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(device_config)
            gpu_id = int(device_config.split(':')[1]) if ':' in device_config else 0
            if gpu_id < torch.cuda.device_count():
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                print(f"Warning: GPU {gpu_id} not available, falling back to cuda:0")
                device = torch.device("cuda:0")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        # 'cpu' or any other value defaults to CPU
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Final device: {device}")
    return device

def get_device_info():
    """Get detailed device information for logging."""
    info = {
        'device_type': None,
        'device_count': 0,
        'device_name': None,
        'memory_total': None,
        'memory_available': None
    }
    
    if torch.cuda.is_available():
        info['device_type'] = 'cuda'
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name()
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        info['memory_total'] = total_memory // (1024**3)  # GB
        info['memory_available'] = (total_memory - allocated_memory) // (1024**3)  # GB
        
    else:
        info['device_type'] = 'cpu'
        info['device_count'] = 1
        info['device_name'] = 'CPU'
    
    return info
