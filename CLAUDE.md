# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a knowledge distillation pipeline for Java unit test assertion generation using a two-stage approach:

1. **Teacher Model Training**: Fine-tune large CodeT5 models on assertion generation
2. **Student Model Distillation**: Train smaller models using advanced multi-component loss functions

### Core Components

- **`knowledge_distillation.py`** - Main student training script with advanced loss functions
- **`train_codet5_assertions.py`** - Teacher model fine-tuning and data generation
- **`config/defaults.py`** - Comprehensive configuration system with 190+ configurable parameters across 13 categories
- **`models/multi_component_loss.py`** - Advanced loss architecture with dynamic weight scheduling
- **`models/loss_functions.py`** - Defines individual loss components (e.g., Focal, JSD, Semantic, PANS).
- **`data/dataset.py`** - Dataset handling with logit compression/decompression
- **`evaluation/evaluators.py`** - Code-specific evaluation metrics used during training.
- **`evaluation/evaluate_assertions.py`** - Script for post-hoc evaluation with teacher/student comparison and comprehensive metrics including Code Quality Score (weighted: 30% CodeBLEU + 20% AST validity + 20% PANS + 15% F1 + 10% semantic similarity + 5% token accuracy).
- **`utils/` directory**: Contains various helper modules, including:
    - `training_utils.py`: Utilities for dynamic hyperparameter scheduling and loss function setup.
    - `logging_utils.py`: Handles logging for training and evaluation.
    - `compress.py`: Manages compression and decompression of teacher logits.

## Key Commands

### Training Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Train teacher model and generate distillation data
python train_codet5_assertions.py --train_data_path data/training.jsonl --val_data_path data/validation.jsonl

# Basic student model training (local)
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --max_train_samples 1000 --max_val_samples 200 \
  --device auto

# Cluster training (DelftBlue/SLURM)
sbatch slurm_scripts/student_training_gpu.sh

# Advanced Trident training (default configuration)
# Note: System auto-detects Trident components and applies TRIDENT_WEIGHT_SCHEDULING
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --loss_components focal jsd semantic \
  --enable_dynamic_weighting \
  --enable_token_weighting \
  --critical_token_weight 2.5 \
  --use_enhanced_metrics \
  --device auto \
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
  --device auto \
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
  --device auto \
  --batch_size 4 --gradient_accumulation_steps 8 --epochs 10

# Post-hoc evaluation with teacher/student comparison
python evaluation/evaluate_assertions.py \
  --teacher_data data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --student_model_path results/run_name/final_model
```

## Loss Function Architecture

The pipeline supports sophisticated multi-component loss functions, now featuring the advanced **Trident** loss function as the default:

### Mathematical Formulation

**Trident Loss Complete Formula:**
```
L_trident = w_focal * L_focal + w_jsd * L_jsd + w_semantic * L_semantic

where:
- L_focal = -α * (1 - p_t)^γ * log(p_t)  [α=0.25, γ=2.0]
- L_jsd = 0.5 * [KL(P||M) + KL(Q||M)] * T^2  [M=(P+Q)/2]
- L_semantic = 1.0 - cosine_similarity(sentence_transform(pred), sentence_transform(ref))
- Dynamic weights: w_i = start_i + progress * (end_i - start_i)
```

**PANS Loss Details:**
```
L_pans = 1 - (∑_{n=1}^{N} ∑_{i,j} exp(-0.1 * |i-j|) * δ(ngram_pred_i, ngram_ref_j)) / max_possible_matches
where δ is the Kronecker delta function for n-gram matching
```

**Temperature Decay:**
```
T(epoch) = T_initial * (T_final/T_initial)^(epoch/total_epochs)
T(epoch) = 4.0 * (1.5/4.0)^(epoch/total_epochs)
```

### Core Loss Functions
- **Traditional**: CE + KL divergence (standard knowledge distillation)
- **Enhanced**: Adds PANS (Position-Aware N-gram Similarity) for code quality
- **AST-Enhanced**: Adds AST validity penalty for syntax correctness
- **Multi-Component**: Full advanced architecture with dynamic weight scheduling
- **Trident** (Multi-Component Default): Advanced focal + JSD + semantic similarity architecture

### Trident Loss Components (Multi-Component Default)
- **Focal Loss**: Replaces standard CE, focuses on hard-to-classify examples
  - Formula: `FL = -α * (1-pt)^γ * log(pt)` where γ=2.0, α=0.25
  - Automatically handles class imbalance and emphasizes difficult examples
- **Jensen-Shannon Divergence (JSD)**: Stable, symmetric alternative to KL divergence
  - Formula: `JSD = 0.5 * (KL(P||M) + KL(Q||M))` where M = (P+Q)/2
  - Numerical stability: eps=1e-8, probability clamping, temperature scaling
- **Semantic Similarity**: Uses sentence transformers for semantic correctness
- **Contrastive Learning**: NEW - InfoNCE loss with CodeBERT embeddings for code understanding
- **Token-Specific Weighting**: NEW - Critical assertion token weighting for improved accuracy
  - Formula: `semantic_loss = 1.0 - cosine_similarity(pred_embedding, ref_embedding)`
  - Uses pre-trained sentence transformer models for meaning preservation

### Dynamic Features
- **Weight Scheduling**: Linear interpolation formula: `weight = start + progress * (end - start)`
  - **Auto-Selection**: System automatically chooses TRIDENT_WEIGHT_SCHEDULING for focal/jsd/semantic components
  - **Trident Scheduling**: focal(0.3→0.3), jsd(0.7→0.35), semantic(0.0→0.35)
  - **Legacy Scheduling**: ce(0.35→0.25), kl(0.6→0.35), pans(0.05→0.25), ast(0.0→0.15)
- **Temperature Decay**: Exponential decay: `T = 4.0 * ((1.5/4.0)^(epoch/total_epochs))`
- **Alpha Adaptation**: Dynamic adjustment based on classification loss convergence
  - Formula: `alpha = max(0.3, 0.5 - (epoch/total_epochs) * 0.2)` when converging

## Data Flow

1. **Input**: JSONL files with focal methods, test methods, and compressed teacher logits
2. **Dataset**: `AssertionDataset` handles on-demand tokenization and logit decompression
3. **Training**: Multi-component loss with gradient accumulation and dynamic scheduling
4. **Evaluation**: Primary metrics (Code Quality Score, Semantic Similarity), plus CodeBLEU, PANS, AST validity, F1, Precision, Recall, Knowledge Retention Score (KRS)
   - **Knowledge Retention Score**: `KRS = 0.6 * output_agreement + 0.4 * performance_ratio`
5. **Output**: Trained models, comprehensive logs, and evaluation reports in `results/`

## Configuration System

- **Defaults**: `config/defaults.py` defines 60+ configurable parameters
- **CLI Arguments**: Rich command-line interface with validation
- **Presets**: Multiple training profiles (quick_start, high_quality, memory_constrained)
- **Dynamic Scheduling**: Unified weight interpolation for all loss components

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
- **Temperature decay**: Exponential decay formula: `T(epoch) = 4.0 * ((1.5/4.0)^(epoch/total_epochs))`
  - Starts at 4.0 for softer probability distributions
  - Decays to 1.5 for sharper focus in final epochs
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

## Prerequisites and Setup

### **System Requirements**
- **Python**: 3.9+ (tested with 3.9-3.11)
- **Memory**: 8GB+ RAM for basic training, 16GB+ recommended for production
- **GPU**: 8GB+ VRAM recommended (CPU training supported but slow)
- **Storage**: 5GB+ for models and data
- **Java**: JDK 8+ for AST validation features

### **Installation Steps**
```bash
# 1. Clone repository
git clone <repository-url>
cd distillation_pipeline

# 2. Install core dependencies
pip install torch transformers datasets sentence-transformers
pip install scikit-learn nltk sacrebleu lz4

# 3. Install optional dependencies for enhanced features
pip install -r requirements.txt  # Full installation

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### **Data Preparation Workflow**

1. **Raw Training Data**: Java methods and test assertions in JSONL format
2. **Teacher Training**: Fine-tune CodeT5 model using `train_codet5_assertions.py`
3. **Distillation Data Generation**: Extract teacher predictions and logits
4. **Student Training**: Train compact model using `knowledge_distillation.py`

```bash
# Example teacher training workflow
python train_codet5_assertions.py \
  --train_data_path data/raw_training.jsonl \
  --val_data_path data/raw_validation.jsonl \
  --output_dir teacher_models/codet5_teacher

# This generates distillation_data_training.jsonl and distillation_data_validation.jsonl
```

## Development Notes

- **Advanced Trident loss** - New default loss function with focal, JSD, and semantic components
- **Unified weight scheduling** - Single `WEIGHT_SCHEDULING` config supports all loss components
- **Sentence transformers** - Required for semantic similarity component (auto-installed)
  - Default model: 'all-MiniLM-L6-v2' for fast inference
  - Alternative models: 'all-mpnet-base-v2' for higher quality
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
├── step_metrics.csv               # NEW: Detailed per-step metrics with gradient norms
├── tensorboard/                   # NEW: TensorBoard logs for visualization
├── distillation_log.txt           # Complete training logs
├── predictions_final.jsonl        # Model predictions
└── metrics_summary_final.json     # Evaluation summary
```

## New Features (PRD v1 Implementation)

### Contrastive Learning
- **CodeBERT Integration**: Frozen microsoft/codebert-base encoder for code embeddings
- **Triplet Sampling**: In-batch sampling (anchor=gold, positive=student, negative=other)
- **InfoNCE Loss**: Stable contrastive objective with temperature=0.1
- **Embedding Cache**: LRU cache with TTL for performance optimization
- **Weight Scheduling**: 0.1 → 0.15 dynamic weighting during training (unified configuration)

### Enhanced Logging & Monitoring
- **Step Metrics CSV**: Detailed per-step logging with gradient norms and system metrics
- **TensorBoard Integration**: Real-time visualization of all scalars and metrics
- **Component Tracking**: Raw vs weighted loss values with metadata
- **Memory Monitoring**: System memory usage and performance tracking
- **Gradient Analysis**: Encoder/decoder gradient norm monitoring

### Semantic Loss Scaling
- **β Parameter**: Configurable semantic loss scaling (default 5.0)
- **Balanced Training**: Ensures semantic loss contributes meaningfully to gradients
- **Analysis Tools**: scripts/analyse_loss_scaling.py for optimization guidance

### Token-Specific Weighting (NEW)
- **Critical Token Database**: 310 curated assertion tokens across 11 categories
- **Automatic Mapping**: Token-to-vocab mapping with multiple fallback strategies
- **Weighted Loss Functions**: Enhanced CE and focal loss with per-token weights
- **Performance Benefits**: +2-5 pp improvement in critical token accuracy
- **CLI Integration**: `--enable_token_weighting --critical_token_weight 2.5`

## Modular Architecture

The codebase follows clean separation of concerns:
- **config/**: Parameter definitions and scheduling presets
- **data/**: Dataset loading with compression utilities
- **models/**: Loss functions and multi-component architecture
- **evaluation/**: Metrics computation and evaluation pipeline
- **utils/**: Logging, training utilities, device management

When extending the system, follow the existing modular patterns and use the comprehensive configuration system in `config/defaults.py`.

## Cluster Compatibility (DelftBlue/SLURM)

The pipeline is fully compatible with HPC clusters including DelftBlue at TU Delft:

### Device Management
- **Automatic GPU Detection**: `--device auto` automatically detects CUDA, falls back to CPU
- **SLURM Integration**: Respects `CUDA_VISIBLE_DEVICES` and SLURM environment variables
- **Multi-GPU Support**: Ready for distributed training setups

### SLURM Job Scripts
Pre-configured SLURM scripts in `slurm_scripts/`:
- `test_cuda.sh` - CUDA compatibility test (30 min, 1 GPU)
- `student_training_gpu.sh` - Main student training (8 hours, 1 GPU)
- `teacher_training_gpu.sh` - Teacher model training (12 hours, 1 GPU) 
- `multi_gpu_training.sh` - Extended training (16 hours, 2 GPUs)

### Usage Example
```bash
# Local development
python knowledge_distillation.py --device auto [args]

# Cluster submission
sbatch slurm_scripts/student_training_gpu.sh

# Monitor job
squeue -u $USER
tail -f logs/slurm-JOBID.out
```

### Cluster Dependencies
Additional monitoring tools in requirements.txt:
- `nvidia-ml-py3` - GPU monitoring on NVIDIA clusters  
- `gpustat` - GPU utilization tracking

See `slurm_scripts/README.md` for detailed cluster setup instructions.