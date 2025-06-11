# SLURM Scripts for DelftBlue Cluster

This directory contains SLURM job scripts for running the knowledge distillation pipeline on the DelftBlue cluster at TU Delft.

## Setup Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure your data files are in the correct locations:
- `data/codet5p-focal-methods/distillation_data_training.jsonl`
- `data/codet5p-focal-methods/distillation_data_validation.jsonl`
- `data/training.jsonl` (for teacher training)
- `data/validation.jsonl` (for teacher training)

### 3. Create Logs Directory
```bash
mkdir -p logs
```

## Available Scripts

### `test_cuda.sh`
Quick test to verify CUDA setup and device detection.
```bash
sbatch slurm_scripts/test_cuda.sh
```
**Resource Requirements:** 1 GPU, 2 CPUs, 8GB RAM, 30 minutes

### `teacher_training_gpu.sh`
Train the CodeT5 teacher model for generating distillation data.
```bash
sbatch slurm_scripts/teacher_training_gpu.sh
```
**Resource Requirements:** 1 GPU, 8 CPUs, 32GB RAM, 12 hours

### `student_training_gpu.sh`
Train the student model using knowledge distillation.
```bash
sbatch slurm_scripts/student_training_gpu.sh
```
**Resource Requirements:** 1 GPU, 8 CPUs, 32GB RAM, 8 hours

### `multi_gpu_training.sh`
Extended training run using multiple GPUs (requires DataParallel support).
```bash
sbatch slurm_scripts/multi_gpu_training.sh
```
**Resource Requirements:** 2 GPUs, 16 CPUs, 64GB RAM, 16 hours

## Script Customization

### Adjusting Resources
Modify these SBATCH parameters based on your needs:
- `--gres=gpu:N` - Number of GPUs (1-4 typically)
- `--cpus-per-task=N` - CPU cores per task
- `--mem=XG` - Memory allocation
- `--time=HH:MM:SS` - Time limit

### Training Parameters
Each script includes common training parameters that you can modify:
- `--batch_size` - Batch size (reduce if memory issues)
- `--gradient_accumulation_steps` - Gradient accumulation
- `--epochs` - Number of training epochs
- `--max_train_samples` - Limit training data size
- `--device auto` - Automatic device detection (uses CUDA if available)

## Key Features for Cluster Compatibility

### 1. Automatic Device Detection
The scripts use `--device auto` which automatically detects and uses available GPUs:
- Respects `CUDA_VISIBLE_DEVICES` environment variable
- Falls back to CPU if no GPU available
- Reports SLURM environment variables

### 2. Environment Variables
The scripts automatically handle:
- `CUDA_VISIBLE_DEVICES` - GPU visibility
- `SLURM_LOCALID` - SLURM local process ID
- `SLURM_JOB_ID` - Job identifier for output directories

### 3. Output Management
- Results saved to timestamped directories: `results/delftblue_*_${SLURM_JOB_ID}/`
- Logs saved to: `logs/slurm-${SLURM_JOB_ID}.out` and `logs/slurm-${SLURM_JOB_ID}.err`

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Job Output
```bash
tail -f logs/slurm-JOBID.out
```

### Cancel Job
```bash
scancel JOBID
```

## GPU Memory Optimization

If you encounter GPU memory issues:

1. **Reduce batch size**: `--batch_size 2`
2. **Increase gradient accumulation**: `--gradient_accumulation_steps 16`
3. **Reduce sequence lengths**: `--max_input_len 256 --max_output_len 128`
4. **Use mixed precision**: The scripts automatically enable FP16 on compatible GPUs

## Troubleshooting

### Common Issues

1. **Module not found errors**: Adjust module load commands in scripts
2. **CUDA out of memory**: Reduce batch size or sequence lengths
3. **Permission denied**: Make scripts executable with `chmod +x slurm_scripts/*.sh`

### Debug Mode
For debugging, you can run the Python commands directly:
```bash
srun --pty --gres=gpu:1 --mem=16G --time=1:00:00 bash
source venv/bin/activate
python knowledge_distillation.py --help
```

## Performance Expectations

### Single GPU Training
- Student model (220M params): ~2-4 hours for 5K samples
- Teacher model (770M params): ~4-8 hours for full dataset

### Multi-GPU Training
- 2 GPUs: ~40-60% speed improvement
- 4 GPUs: Limited by communication overhead, ~100-150% improvement

## Contact

For DelftBlue-specific issues, consult the TU Delft HPC documentation or contact cluster support.