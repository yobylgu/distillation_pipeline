# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a knowledge distillation pipeline for Java unit test assertion generation using a two-stage approach:

1. **Teacher Model Training**: Fine-tune large CodeT5 models on assertion generation
2. **Student Model Distillation**: Train smaller models using advanced multi-component loss functions

### Core Components

- **`knowledge_distillation.py`** - Main student training script with advanced loss functions
- **`train_codet5_assertions.py`** - Teacher model fine-tuning and data generation
- **`config/defaults.py`** - Comprehensive configuration system with 400+ parameters
- **`models/multi_component_loss.py`** - Advanced loss architecture with dynamic weight scheduling
- **`data/dataset.py`** - Dataset handling with logit compression/decompression
- **`evaluation/evaluators.py`** - Code-specific evaluation metrics

## Key Commands

### Training Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Train teacher model and generate distillation data
python train_codet5_assertions.py --train_data_path data/training.jsonl --val_data_path data/validation.jsonl

# Basic student model training
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --max_train_samples 1000 --max_val_samples 200

# Advanced Trident training (default configuration)
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --enable_dynamic_weighting \
  --use_enhanced_metrics \
  --batch_size 4 --gradient_accumulation_steps 8 --epochs 10

# Extended Trident training (7-8 hour runs)
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --max_train_samples 5500 \
  --max_val_samples 550 \
  --batch_size 4 \
  --epochs 8 \
  --gradient_accumulation_steps 4 \
  --lr 5e-05 \
  --warmup_steps 275 \
  --weight_decay 0.01 \
  --alpha 0.5 \
  --temperature 4.0 \
  --output_dir results/extended_trident_7h \
  --loss_function multi_component \
  --loss_components focal jsd semantic \
  --dropout_rate 0.1 \
  --enable_dynamic_weighting \
  --use_enhanced_metrics

# Legacy multi-component training
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --loss_components ce kl pans ast \
  --enable_dynamic_weighting \
  --use_enhanced_metrics \
  --batch_size 4 --gradient_accumulation_steps 8 --epochs 10

# Post-hoc evaluation
python evaluation/evaluate_assertions.py \
  --teacher_data data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --student_model_path results/run_name/final_model
```

## Loss Function Architecture

The pipeline supports sophisticated multi-component loss functions, now featuring the advanced **Trident** loss function as the default:

### Core Loss Functions
- **Traditional**: CE + KL divergence (standard knowledge distillation)
- **Enhanced**: Adds PANS (Position-Aware N-gram Similarity) for code quality
- **AST-Enhanced**: Adds AST validity penalty for syntax correctness
- **Multi-Component**: Full advanced architecture with dynamic weight scheduling
- **Trident** (Default): Advanced focal + JSD + semantic similarity architecture

### Trident Loss Components (Default)
- **Focal Loss**: Replaces standard CE, focuses on hard-to-classify examples
- **Jensen-Shannon Divergence (JSD)**: Stable, symmetric alternative to KL divergence
- **Semantic Similarity**: Uses sentence transformers for semantic correctness
- **PANS**: Position-Aware N-gram Similarity for code quality
- **AST**: Abstract Syntax Tree validity penalty

### Dynamic Features
- **Weight Scheduling**: Linear interpolation of loss component weights during training
- **Temperature Decay**: Gradual annealing of distillation temperature 
- **Alpha Adaptation**: Automatic tuning based on validation performance

## Data Flow

1. **Input**: JSONL files with focal methods, test methods, and compressed teacher logits
2. **Dataset**: `AssertionDataset` handles on-demand tokenization and logit decompression
3. **Training**: Multi-component loss with gradient accumulation and dynamic scheduling
4. **Evaluation**: Code-specific metrics (BLEU, PANS, AST validity, F1)
5. **Output**: Trained models, comprehensive logs, and evaluation reports in `results/`

## Configuration System

- **Defaults**: `config/defaults.py` defines 400+ configurable parameters
- **CLI Arguments**: Rich command-line interface with validation
- **Presets**: Multiple training profiles (quick_start, high_quality, memory_constrained)
- **Dynamic Scheduling**: Configurable weight interpolation strategies

## Parameter Guidance

### Warmup Steps Configuration
The system uses both `warmup_steps` and `warmup_ratio` parameters:
- **Direct warmup**: Use `--warmup_steps` for explicit step count (e.g., 275 steps)
- **Proportional warmup**: Use `--warmup_ratio` for percentage of total steps (e.g., 0.1 = 10%)
- **Calculation**: When using ratio, actual steps = `warmup_ratio * total_training_steps`
- **Total steps**: Calculated as `(num_samples / batch_size) * epochs / gradient_accumulation_steps`
- **Recommendation**: Use explicit `warmup_steps` for predictable scheduling

### Alpha Parameter Behavior
Controls the balance between student and teacher losses:
- **Default**: 0.5 (equal weighting between student CE loss and distillation loss)
- **Range**: 0.0 to 1.0
- **Dynamic behavior**: When `enable_dynamic_weighting=True`, alpha is automatically adjusted:
  - Starts at specified alpha value
  - Gradually increases student loss weight during training
  - Adaptive adjustment based on validation performance
- **Static behavior**: When dynamic weighting disabled, alpha remains constant

### Temperature Guidelines for Trident Loss
Temperature controls the softness of probability distributions:
- **Default**: 4.0 (optimal for Trident loss with focal/JSD components)
- **Traditional KL**: Use 3.0-5.0
- **Focal + JSD**: Use 4.0-6.0 (higher temperatures work better)
- **Temperature decay**: Automatically enabled, gradually reduces from initial value
- **Effect**: Higher temperatures create softer distributions, lower temperatures sharpen them

### Learning Rate and Scheduling
- **Base LR**: 5e-5 works well for CodeT5+ models
- **Warmup**: Essential for stable training, use 5-10% of total steps
- **Weight decay**: 0.01 provides good regularization
- **Gradient clipping**: Automatically enabled at 1.0 to prevent instability

### Batch Size and Memory Management
- **Effective batch size**: `batch_size * gradient_accumulation_steps`
- **Memory optimization**: Use smaller batch_size with higher gradient_accumulation_steps
- **Recommendation**: batch_size=4, gradient_accumulation_steps=4-8
- **Large datasets**: Increase gradient_accumulation_steps rather than batch_size

## Development Notes

- **Advanced Trident loss** - New default loss function with focal, JSD, and semantic components
- **Sentence transformers** - Required for semantic similarity component (auto-loaded when needed)
- **No formal build system** - manual dependency management with requirements.txt
- **No testing infrastructure** - missing unit tests and CI/CD
- **Comprehensive logging** - `DistillationLogger` with CSV/JSON export
- **Memory optimization** - Teacher logit compression using LZ4
- **Production features** - Early stopping, gradient clipping, error handling

## Results Structure

Training outputs are organized in timestamped directories under `results/`:
```
results/YYYY-MM-DD_HH-MM-SS_model-name/
├── final_model/                    # Saved student model
├── best_model/                     # Best checkpoint (early stopping)
├── training_metrics.csv            # Step-by-step training metrics
├── distillation_log.txt           # Complete training logs
├── predictions_final.jsonl        # Model predictions
└── metrics_summary_final.json     # Evaluation summary
```

## Modular Architecture

The codebase follows clean separation of concerns:
- **config/**: Parameter definitions and scheduling presets
- **data/**: Dataset loading with compression utilities
- **models/**: Loss functions and multi-component architecture
- **evaluation/**: Metrics computation and evaluation pipeline
- **utils/**: Logging, training utilities, device management

When extending the system, follow the existing modular patterns and use the comprehensive configuration system in `config/defaults.py`.