# Knowledge Distillation Pipeline for Java Unit Test Assertion Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Advanced knowledge distillation pipeline for automatically generating Java unit test assertions using a two-stage approach:

1. **Teacher Model**: Fine-tuned CodeT5 model on assertion generation
2. **Student Model**: Smaller model trained via knowledge distillation with advanced loss functions

## Key Features

- **Trident Loss**: Focal loss + Jensen-Shannon divergence + semantic similarity (default)
- **Multi-Component Losses**: Traditional, Enhanced (PANS), AST-aware options
- **Token-Specific Weighting**: Critical assertion token weighting for improved accuracy
- **Memory-Efficient Training**: Epoch sampling for resource-constrained environments
- **Dynamic Weight Scheduling**: Configurable linear interpolation for loss components
- **Production Ready**: Comprehensive logging, error handling, cluster compatibility

## Project Structure

```
distillation_pipeline/
├── knowledge_distillation.py      # Main training script
├── train_codet5_assertions.py     # Teacher model fine-tuning
├── config/
│   ├── defaults.py                # Configuration parameters
│   └── critical_tokens.py         # Critical assertion tokens
├── data/
│   └── dataset.py                 # Dataset handling
├── evaluation/
│   ├── evaluate_assertions.py     # Post-hoc evaluation
│   └── evaluators.py              # Evaluation metrics
├── models/
│   ├── loss_functions.py          # Loss components
│   └── multi_component_loss.py    # Multi-component loss
├── utils/                         # Helper utilities
└── slurm_scripts/                 # Cluster job scripts
```

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **GPU**: 8GB+ VRAM recommended (CPU supported)
- **Java JDK 8+** for AST validation

### Installation

```bash
# Clone repository
git clone <repository-url>
cd distillation_pipeline

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers; print('✓ Installation successful')"
```

### Data Preparation

Train teacher model and generate distillation data:

```bash
# Train teacher model
python train_codet5_assertions.py \
  --train_data_path data/raw_training.jsonl \
  --val_data_path data/raw_validation.jsonl \
  --output_dir teacher_models/codet5_teacher

# This generates distillation_data_training.jsonl and distillation_data_validation.jsonl
```

**Expected JSONL format:**
```json
{
  "focal_method": "public boolean isEmpty() { return size == 0; }",
  "test_method": "@Test public void testIsEmpty() { ... }",
  "assertion": "assertTrue(list.isEmpty());",
  "teacher_logits_compressed": "<compressed-logits>",
  "teacher_prediction": "assertTrue(list.isEmpty());"
}
```

### Basic Training

```bash
# Default Trident loss training (recommended)
python knowledge_distillation.py \
  --train_data_path data/distillation_data_training.jsonl \
  --val_data_path data/distillation_data_validation.jsonl \
  --max_train_samples 1000 \
  --batch_size 4 \
  --epochs 5 \
  --device auto

# Memory-efficient training (for limited RAM)
python knowledge_distillation.py \
  --train_data_path data/distillation_data_training.jsonl \
  --val_data_path data/distillation_data_validation.jsonl \
  --max_train_samples 500 \
  --enable_epoch_sampling \
  --device auto

# Cluster training
sbatch slurm_scripts/student_training_gpu.sh
```

### Advanced Training

```bash
# Production configuration with all features
python knowledge_distillation.py \
  --train_data_path data/distillation_data_training.jsonl \
  --val_data_path data/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --loss_components focal jsd semantic \
  --enable_dynamic_weighting \
  --enable_token_weighting \
  --critical_token_weight 2.5 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 10 \
  --device auto
```

## Loss Functions

| Function | Components | Use Case |
|----------|------------|----------|
| `traditional` | CE + KL | Baseline distillation |
| `multi_component` | Focal + JSD + Semantic | **Production (default)** |
| `multi_component` | CE + KL + PANS + AST | Legacy with code quality |

**Trident Loss (default)**: Combines focal loss, Jensen-Shannon divergence, and semantic similarity with dynamic weight scheduling.

## Configuration

### Key Parameters

- `--loss_function`: Choose loss type (`traditional`, `multi_component`)
- `--loss_components`: Components for multi_component (`focal jsd semantic` or `ce kl pans ast`) 
- `--enable_dynamic_weighting`: Enable weight scheduling
- `--enable_token_weighting`: Focus on critical assertion tokens
- `--enable_epoch_sampling`: Memory-efficient training for large datasets
- `--device auto`: Automatic GPU/CPU detection

For all options: `python knowledge_distillation.py --help`

## Evaluation

```bash
# Post-hoc evaluation comparing teacher and student models
python evaluation/evaluate_assertions.py \
  --teacher_data data/distillation_data_validation.jsonl \
  --student_model_path results/run_name/final_model
```

**Key Metrics:**
- **Code Quality Score**: Weighted combination of CodeBLEU, AST validity, semantic similarity
- **Knowledge Retention Score**: Measures distillation effectiveness 
- **Semantic Similarity**: Meaning preservation using sentence transformers

## Memory-Efficient Training

For resource-constrained environments (e.g., MacBook Pro 16GB):

```bash
# Use epoch sampling to avoid loading full dataset
python knowledge_distillation.py \
  --train_data_path data/distillation_data_training.jsonl \
  --val_data_path data/distillation_data_validation.jsonl \
  --max_train_samples 1000 \
  --enable_epoch_sampling \
  --seed 42 \
  --device auto
```

**Features:**
- **Epoch Sampling**: Loads different random subset each epoch
- **Memory Optimization**: Never loads full dataset into memory
- **Reproducible**: Uses configurable seeds for consistent experiments

## Troubleshooting

**Common Issues:**
- **Out of Memory**: Reduce `--batch_size` to 2, increase `--gradient_accumulation_steps`
- **Import Errors**: Run `pip install -r requirements.txt`
- **Data Format Error**: Ensure data has compressed teacher logits
- **Slow Training**: Disable `--use_enhanced_metrics` during training
- **NaN Loss**: Reduce learning rate to `--lr 1e-5`

**Hardware Recommendations:**
- 8GB GPU: `--batch_size 2 --gradient_accumulation_steps 8`
- 16GB GPU: `--batch_size 4 --gradient_accumulation_steps 8` 
- CPU only: `--batch_size 1 --gradient_accumulation_steps 16`

## Documentation

- **CLAUDE.md**: Detailed implementation notes and architecture
- **config/defaults.py**: All configuration parameters
- **slurm_scripts/**: Cluster job examples

Run `python knowledge_distillation.py --help` for all options.
