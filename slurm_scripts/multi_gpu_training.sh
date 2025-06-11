#!/bin/bash
#SBATCH --job-name=distillation_multi_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# DelftBlue SLURM job script for multi-GPU knowledge distillation training
# Usage: sbatch slurm_scripts/multi_gpu_training.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs available: $(nvidia-smi --list-gpus | wc -l)"

# Load modules if needed (adjust for DelftBlue environment)
# module load Python/3.9.6-GCCcore-11.2.0
# module load CUDA/11.7.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up Python environment
source venv/bin/activate  # Adjust path to your virtual environment

# Print GPU information
echo "GPU Information:"
nvidia-smi

# For multi-GPU training, you can use DataParallel or DistributedDataParallel
# This example uses single-process with DataParallel (simpler setup)
# For true distributed training, you'd need to modify the training script

python knowledge_distillation.py \
    --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
    --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
    --max_train_samples 10000 \
    --max_val_samples 1000 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --epochs 10 \
    --lr 5e-05 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --alpha 0.5 \
    --temperature 4.0 \
    --device auto \
    --output_dir results/delftblue_multi_gpu_${SLURM_JOB_ID} \
    --loss_function multi_component \
    --loss_components focal jsd semantic contrastive \
    --dropout_rate 0.1 \
    --enable_dynamic_weighting \
    --enable_token_weighting \
    --critical_token_weight 2.5 \
    --use_enhanced_metrics

echo "Job finished at: $(date)"