#!/bin/bash
#SBATCH --job-name=codet5_teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# DelftBlue SLURM job script for CodeT5 teacher model training
# Usage: sbatch slurm_scripts/teacher_training_gpu.sh

echo "Job started at: $(date)"
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

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Run the teacher model training
python train_codet5_assertions.py \
    --train_data_path data/training.jsonl \
    --val_data_path data/validation.jsonl \
    --model_name Salesforce/codet5p-770m \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 5 \
    --lr 2e-5 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --output_dir results/teacher_delftblue_${SLURM_JOB_ID} \
    --max_input_len 512 \
    --max_output_len 256 \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 100

echo "Job finished at: $(date)"