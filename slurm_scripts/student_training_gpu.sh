#!/bin/bash
#SBATCH --job-name=distillation_student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# DelftBlue SLURM job script for knowledge distillation student training
# Usage: sbatch slurm_scripts/student_training_gpu.sh

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

# Run the knowledge distillation training
python knowledge_distillation.py \
    --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
    --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
    --max_train_samples 5500 \
    --max_val_samples 550 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --epochs 8 \
    --lr 5e-05 \
    --warmup_steps 275 \
    --weight_decay 0.01 \
    --alpha 0.5 \
    --temperature 4.0 \
    --device auto \
    --output_dir results/delftblue_gpu_${SLURM_JOB_ID} \
    --loss_function multi_component \
    --loss_components focal jsd semantic \
    --dropout_rate 0.1 \
    --enable_dynamic_weighting \
    --enable_token_weighting \
    --critical_token_weight 2.5 \
    --use_enhanced_metrics \
    --fp16

echo "Job finished at: $(date)"