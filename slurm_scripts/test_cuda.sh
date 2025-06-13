#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cuda_test-%j.out
#SBATCH --error=logs/cuda_test-%j.err

# DelftBlue SLURM job script for testing CUDA setup
# Usage: sbatch slurm_scripts/test_cuda.sh

echo "CUDA Test Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load modules if needed (adjust for DelftBlue environment)
# module load Python/3.9.6-GCCcore-11.2.0
# module load CUDA/11.7.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up Python environment
source venv/bin/activate  # Adjust path to your virtual environment

# Print system information
echo "=== System Information ==="
nvidia-smi
echo ""

echo "=== Python CUDA Test ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=== Device Utils Test ==="
python -c "
from utils.device_utils import setup_device, get_device_info
device = setup_device('auto')
info = get_device_info()
print(f'Selected device: {device}')
print(f'Device info: {info}')

# Test basic tensor operations
import torch
x = torch.randn(10, 10).to(device)
y = torch.randn(10, 10).to(device)
z = torch.mm(x, y)
print(f'Matrix multiplication test passed on {device}')
"

echo "CUDA test completed at: $(date)"