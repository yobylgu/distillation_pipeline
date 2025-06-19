# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a knowledge distillation pipeline for Java unit test assertion generation using a two-stage approach:

1. **Teacher Model Training**: Fine-tune large CodeT5 models on assertion generation
2. **Student Model Distillation**: Train smaller models using advanced multi-component loss functions

### Core Components

- **`knowledge_distillation.py`** - Main student training script with advanced loss functions
- **`train_codet5_assertions.py`** - Teacher model fine-tuning and data generation
- **`config/defaults.py`** - Comprehensive configuration system with 100+ configurable parameters across 13 categories
- **`models/multi_component_loss.py`** - Advanced loss architecture with dynamic weight scheduling
- **`models/loss_functions.py`** - Defines individual loss components (e.g., Focal, JSD, Semantic, PANS).
- **`data/dataset.py`** - Dataset handling with logit compression/decompression
- **`evaluation/evaluators.py`** - Code-specific evaluation metrics used during training.
- **`evaluation/evaluate_assertions.py`** - Script for post-hoc evaluation with teacher/student comparison and comprehensive metrics including Code Quality Score (optimized for test assertions: 25% CodeBLEU + 25% CodeSearchNet semantic similarity + 20% AST validity + 20% token similarity + 10% token accuracy).
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

# Memory-efficient training with epoch sampling (recommended for MacBook Pro 16GB)
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --max_train_samples 1000 --max_val_samples 200 \
  --enable_epoch_sampling \
  --seed 42 --sampling_seed 42 \
  --device auto

# Cluster training (DelftBlue/SLURM)
sbatch slurm_scripts/student_training_gpu.sh

# Advanced Trident training (default configuration) - OPTIMIZED
# Note: System auto-detects Trident components and applies TRIDENT_WEIGHT_SCHEDULING
# NEW: Includes AMP (--fp16) for 1.5-3x speedup on modern GPUs
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --loss_components focal jsd semantic \
  --enable_dynamic_weighting \
  --enable_token_weighting \
  --critical_token_weight 2.5 \
  --use_enhanced_metrics \
  --seed 42 \
  --device auto \
  --batch_size 4 --gradient_accumulation_steps 8 --epochs 10 \
  --fp16

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
  --use_enhanced_metrics \
  --seed 42 \
  --fp16

# Legacy multi-component training
python knowledge_distillation.py \
  --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
  --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
  --loss_function multi_component \
  --loss_components ce kl pans ast \
  --enable_dynamic_weighting \
  --use_enhanced_metrics \
  --seed 42 \
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
   - **Standard Mode**: Loads fixed subset at initialization (sequential or up to max_samples)
   - **Epoch Sampling Mode**: Loads different random subset each epoch for memory efficiency
3. **Training**: Multi-component loss with gradient accumulation and dynamic scheduling
4. **Evaluation**: Primary metrics (Code Quality Score, Semantic Similarity), plus CodeBLEU, PANS, AST validity, F1, Precision, Recall, Knowledge Retention Score (KRS)
   - **Knowledge Retention Score**: `KRS = 0.6 * output_agreement + 0.4 * performance_ratio`
5. **Output**: Trained models, comprehensive logs, and evaluation reports in `results/`

## Memory-Efficient Epoch Sampling

New feature for resource-constrained environments (e.g., MacBook Pro 16GB):

### **EpochSamplingDataset**
- **Memory Optimization**: Loads only `max_train_samples` per epoch, clears after each epoch
- **Random Sampling**: Different random subset from full dataset each epoch
- **Reproducible**: Uses configurable seed + epoch offset for consistent experiments
- **Coverage Tracking**: Logs memory usage, sampling statistics, and data coverage
- **CLI Integration**: `--enable_epoch_sampling --sampling_seed 42`

### **Benefits for M1 MacBook Pro**
- **RAM Efficiency**: Never loads full 20k dataset, only subset in memory
- **Wider Exposure**: Sees more diverse data across epochs vs. fixed subset
- **Better Generalization**: Random sampling reduces overfitting to specific samples
- **Flexible Training**: Adjust `max_train_samples` based on available memory

### **Usage Pattern**
```bash
# Memory-efficient training for large datasets
python knowledge_distillation.py \
  --train_data_path data/large_dataset.jsonl \
  --max_train_samples 2000 \
  --enable_epoch_sampling \
  --seed 42 \
  --epochs 10 \
  --device auto
```

## Reproducibility and Seeding

The pipeline provides comprehensive seeding for fully reproducible training:

### **Seeding System**
- **Global Seed**: `--seed` parameter controls all random number generators
- **Automatic Seeding**: Python random, NumPy, PyTorch CPU/GPU generators
- **CUDA Determinism**: Enables deterministic CUDA operations when available
- **Data Shuffling**: Consistent data ordering across identical runs
- **Worker Processes**: Environment variables set for reproducible data loading

### **Usage Examples**
```bash
# Use default seed (42) for reproducible training
python knowledge_distillation.py --seed 42 [other args...]

# Use custom seed for different random behavior
python knowledge_distillation.py --seed 123 [other args...]

# Epoch sampling with both global and sampling seeds
python knowledge_distillation.py --seed 42 --sampling_seed 42 --enable_epoch_sampling [other args...]
```

### **Seed Parameters**
- **`--seed`**: Global random seed for all operations (default: 42)
- **`--sampling_seed`**: Specific seed for epoch sampling when `--enable_epoch_sampling` is used
- **Consistent Results**: Same seed + same config = identical metrics across runs

## Configuration System

- **Defaults**: `config/defaults.py` defines 100+ configurable parameters
- **CLI Arguments**: Rich command-line interface with validation
- **Presets**: Multiple training profiles (quick_start, high_quality, memory_constrained)
- **Dynamic Scheduling**: Unified weight interpolation for all loss components

## Parameter Guidance

### Warmup Steps Configuration
The system uses both `warmup_steps` and `warmup_ratio` parameters:
- **Direct warmup**: Use `--warmup_steps` for explicit step count (e.g., 275 steps)
- **Proportional warmup**: Use `--warmup_ratio` for percentage of total steps (e.g., 0.15 = 15%)
- **Calculation**: When using ratio, actual steps = `warmup_ratio * total_training_steps`
- **Total steps**: Calculated as `(num_samples / batch_size) * epochs / gradient_accumulation_steps`
- **Recommendation**: Use explicit `warmup_steps` for predictable scheduling (default is 15% of total steps)

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
- **Base LR**: 5e-5 works well for CodeT5+ models (updated default)
- **Warmup**: Essential for stable training, use 15% of total steps (updated default)
- **Weight decay**: 0.01 provides good regularization
- **Gradient clipping**: Automatically enabled at 1.0 to prevent instability
- **LR Scheduling**: Linear decay with minimum floor at 1e-5 (20% of initial LR)

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
  - **CodeSearchNet model**: 'embaas/codesearchnet-minilm-l6' for code-aware semantic similarity (used in both training and evaluation)
  - Legacy model: 'all-MiniLM-L6-v2' for general text similarity (replaced)
  - Alternative models: 'all-mpnet-base-v2' for higher quality
- **No formal build system** - manual dependency management with requirements.txt
- **No testing infrastructure** - missing unit tests and CI/CD
- **Comprehensive logging** - `DistillationLogger` with CSV/JSON export
- **Memory optimization** - Teacher logit compression using LZ4
- **Production features** - Early stopping, gradient clipping, error handling

## Performance Optimizations (NEW)

The pipeline now includes several key performance optimizations for modern GPU training:

### **Automatic Mixed Precision (AMP)**
- **Implementation**: `torch.cuda.amp.GradScaler` and `autocast()` context manager
- **Activation**: Use `--fp16` flag to enable mixed precision training
- **Benefits**: 1.5-3x speedup on modern GPUs (A100, V100, RTX series) + reduced memory usage
- **Compatibility**: Automatically falls back to FP32 if CUDA unavailable

### **Optimized Data Loading**
- **Workers**: `DEFAULT_NUM_WORKERS = 8` (matches SLURM `--cpus-per-task=8`)
- **Pinned Memory**: `pin_memory=True` for faster CPU-GPU transfers
- **Previous Bottleneck**: Single-threaded loading (num_workers=0) limited GPU utilization
- **Impact**: Eliminates data loading bottleneck for high-throughput GPUs

### **Performance Commands**
```bash
# High-performance cluster training (recommended)
python knowledge_distillation.py \
  --train_data_path data/training.jsonl \
  --val_data_path data/validation.jsonl \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --fp16 \
  --device auto

# Memory-efficient local training (M1 MacBook Pro compatible)
python knowledge_distillation.py \
  --max_train_samples 1000 \
  --enable_epoch_sampling \
  --device auto \
  --num_workers 2
  
# For M1 MacBook Pro (CPU only, no --fp16)
python knowledge_distillation.py \
  --max_train_samples 500 \
  --enable_epoch_sampling \
  --device cpu \
  --num_workers 2 \
  --batch_size 2
```

### **Expected Performance Gains**
- **A100/V100 GPU (with --fp16)**: 2-4x faster training vs. previous implementation  
- **M1 MacBook Pro (CPU)**: Modest improvements from optimized data loading
- **Memory Usage**: 30-40% reduction with FP16 on CUDA, enabling larger batch sizes
- **Data Loading**: 2-4x faster parallel loading vs. single-threaded baseline
- **Compatibility**: Works seamlessly on CPU, CUDA, and mixed environments

### **M1 MacBook Pro Optimization Guide**
The pipeline is fully compatible with M1 MacBook Pro (16GB RAM). Key recommendations:

**Optimal Settings for M1:**
```bash
python knowledge_distillation.py \
  --max_train_samples 500 \
  --max_val_samples 50 \
  --enable_epoch_sampling \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_workers 2 \
  --device auto \
  --epochs 3
```

**Memory Management:**
- Use `--enable_epoch_sampling` to avoid loading full dataset
- Keep `--max_train_samples` ≤ 1000 for 16GB RAM
- Set `--num_workers 2` (conservative for M1's CPU cores)
- Avoid `--fp16` (not supported on CPU)

**Performance Notes:**
- Training on M1 CPU is slower but functional for development/testing
- Use cluster with CUDA GPUs for production training
- Data loading optimizations still provide 2x speedup on M1

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