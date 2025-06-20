"""
Default configuration settings for the distillation pipeline.
"""

DEFAULT_MODEL_NAME = 'Salesforce/codet5p-220m'
DEFAULT_MAX_INPUT_LEN = 512
DEFAULT_MAX_OUTPUT_LEN = 128
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_ALPHA = 0.5
DEFAULT_TEMPERATURE = 4.0  # Changed to match documentation and examples

# Data processing defaults
DEFAULT_MAX_TRAIN_SAMPLES = 100
DEFAULT_MAX_VAL_SAMPLES = 50
DEFAULT_OUTPUT_DIR = 'results/distillation_run'

# Default configurations for overfitting prevention
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001
DEFAULT_VALIDATION_FREQUENCY = 1  # Validate every N epochs

# Training optimization defaults  
DEFAULT_WARMUP_STEPS = 0  # 0 means auto-calculate as 15% of total steps
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_WARMUP_RATIO = 0.15  # 15% of total steps for warmup
DEFAULT_ADAM_EPSILON = 1e-8

# Loss function defaults
DEFAULT_LOSS_FUNCTION = 'multi_component'  # Changed to match documentation
DEFAULT_LOSS_COMPONENTS = ['focal', 'jsd', 'semantic']  # Trident components
DEFAULT_ENABLE_DYNAMIC_WEIGHTING = True

# DataLoader defaults
DEFAULT_NUM_WORKERS = 4  # Conservative default that works on most systems (including M1 MacBook Pro)
DEFAULT_PIN_MEMORY = True  # Already enabled in the code, documenting here
DEFAULT_SHUFFLE_TRAIN = True
DEFAULT_SHUFFLE_EVAL = False

# Mixed precision defaults
DEFAULT_FP16 = False  # Enable with --fp16 flag for significant performance gains on modern GPUs

# Reproducibility defaults
DEFAULT_SEED = 42  # Random seed for reproducible training

# Loss function configurations
LOSS_FUNCTION_CHOICES = ['traditional', 'enhanced', 'ast_enhanced', 'multi_component']
LOSS_COMPONENT_CHOICES = ['ce', 'kl', 'pans', 'ast', 'focal', 'jsd', 'semantic', 'contrastive']

# Contrastive learning parameters (Task 2 - PRD v1)
DEFAULT_CONTRASTIVE_WEIGHT = 0.1  # Initial contrastive weight
DEFAULT_CONTRASTIVE_TEMPERATURE = 0.1  # InfoNCE temperature

# Default weights for multi-component loss
DEFAULT_LOSS_WEIGHTS = {
    'ce': 0.45,
    'kl': 0.4,
    'pans': 0.1,
    'ast': 0.05,
    'focal': 0.45,  # Similar weight to CE for basic classification
    'jsd': 0.4,     # Similar weight to KL for knowledge distillation
    'semantic': 0.15,  # Moderate weight for semantic similarity
    'contrastive': 0.1  # Initial contrastive weight (PRD v1)
}

# Enhanced loss weights
DEFAULT_PANS_WEIGHT = 0.12
DEFAULT_AST_WEIGHT = 0.2

# Running Average Normalization for balanced gradient magnitudes
# Normalizes each loss component by its running average to ensure similar scales
DEFAULT_RUNNING_AVG_MOMENTUM = 0.99  # Momentum for exponential moving average
DEFAULT_ENABLE_LOSS_NORMALIZATION = True  # Enable running average normalization
DEFAULT_LOSS_NORM_WARMUP_STEPS = 100  # Steps to accumulate before applying normalization

# Running average normalization configuration
LOSS_NORMALIZATION_PARAMS = {
    'enabled': DEFAULT_ENABLE_LOSS_NORMALIZATION,  # Enable running average normalization
    'momentum': DEFAULT_RUNNING_AVG_MOMENTUM,  # Exponential moving average momentum (0.9-0.999)
    'min_scale': 1e-8,  # Minimum running average to prevent division by zero
    'warmup_steps': DEFAULT_LOSS_NORM_WARMUP_STEPS,  # Steps to accumulate before applying normalization
    'normalize_components': ['focal', 'jsd', 'semantic', 'ce', 'kl', 'pans', 'ast', 'contrastive']  # Which components to normalize
}

# Dynamic weight scheduling for multi-component loss (UNIFIED SCHEDULING)
# Linear interpolation strategy: weight = start + progress * (end - start)
# where progress goes from 0.0 (start of training) to 1.0 (end of training)
# All loss components are normalized, so exact values matter less than relative proportions
WEIGHT_SCHEDULING = {
    # Legacy components (CE + KL + PANS + AST)
    'ce': {
        'start': 0.35,  # Lower CE, prioritizing knowledge transfer
        'end': 0.25     # Further reduced to make room for code components
    },
    'kl': {
        'start': 0.6,   # High KL priority throughout training
        'end': 0.35     # Still prioritized but allows growth of code components
    },
    'pans': {
        'start': 0.05,  # Small initial PANS
        'end': 0.25     # Major PANS increase for code quality
    },
    'ast': {
        'start': 0.0,   # No initial AST penalty
        'end': 0.15     # Moderate AST growth for syntax
    },
    
    # Trident components (Focal + JSD + Semantic)
    'focal': {
        'start': 0.3,   # Strong focal loss start for hard example focus
        'end': 0.25     # Reduced to balance with semantic and contrastive learning
    },
    'jsd': {
        'start': 0.6,   # High JSD priority for stable knowledge transfer
        'end': 0.3     # Maintained but allows semantic and contrastive growth
    },
    'semantic': {
        'start': 0.1,  # Low semantic start to establish foundations
        'end': 0.3     # Major semantic growth for advanced understanding
    },
    'contrastive': {
        'start': 0.1,   # PRD v1: Start with moderate contrastive weight
        'end': 0.15     # PRD v1: Increase to strengthen contrastive learning
    }
}


# Weight normalization settings
WEIGHT_NORMALIZATION = {
    'enabled': True,   # Whether to normalize weights to sum to 1.0
    'min_weight': 0.01,  # Minimum weight for any component
    'max_weight': 0.8    # Maximum weight for any component
}

# ============================================================================
# TRAINING OPTIMIZATION PARAMETERS
# ============================================================================

# Core Training Configuration
TRAINING_PARAMS = {
    'batch_size': 4,            # Tune: 2-16 depending on GPU memory. Higher = more stable gradients but needs more memory
    'gradient_accumulation_steps': 4,  # Tune: 1-8 to simulate larger batch sizes. Use when batch_size is limited by memory
    'learning_rate': 5e-5,      # Tune: 1e-6 to 1e-3. Start with 5e-5, increase for faster convergence, decrease for stability
    'warmup_steps': 500,        # Tune: 0-2000. Gradual LR increase prevents early instability. Use 10-20% of total steps
    'weight_decay': 0.01,       # Tune: 0.0-0.3. L2 regularization. Increase to reduce overfitting, decrease for underfitting
    'max_grad_norm': 1.0,       # Tune: 0.5-5.0. Gradient clipping. Lower values = more stable training
    'epochs': 5,                # Tune: 3-20. Monitor validation loss to avoid overfitting
    'save_strategy': 'epoch',   # Options: 'epoch', 'steps', 'no'. How often to save checkpoints
    'eval_strategy': 'epoch',   # Options: 'epoch', 'steps', 'no'. How often to evaluate
    'logging_steps': 10,        # Tune: 1-100. How often to log training metrics
}

# Optimizer Configuration  
OPTIMIZER_PARAMS = {
    'optimizer_type': 'adamw',  # Options: 'adamw', 'adam', 'sgd'. AdamW generally best for transformers
    'adam_epsilon': 1e-8,       # Tune: 1e-10 to 1e-6. Numerical stability. Decrease if training unstable
    'adam_beta1': 0.9,          # Tune: 0.8-0.95. Momentum for first moment. Higher = more momentum
    'adam_beta2': 0.999,        # Tune: 0.99-0.9999. Momentum for second moment. Higher = more stability
    'lr_scheduler_type': 'linear',  # Options: 'linear', 'cosine', 'constant'. Linear good for distillation
}

# Learning Rate Scheduling
LR_SCHEDULE_PARAMS = {
    'warmup_ratio': 0.15,        # Tune: 0.0-0.3. Fraction of training for warmup. Higher = more gradual start
    'cosine_restarts': False,   # Enable cosine annealing with restarts for better convergence
    'num_cycles': 1,            # Tune: 1-5. Number of cosine cycles if using cosine scheduler
    'lr_end_factor': 0.2,       # Tune: 0.01-0.5. Final LR as fraction of initial LR (ensures min LR of 1e-5)
}

# ============================================================================
# MODEL CONFIGURATION PARAMETERS  
# ============================================================================

MODEL_PARAMS = {
    'model_name': 'Salesforce/codet5-small',  # Teacher model for distillation
    'student_model': None,      # If None, uses same architecture as teacher but smaller
    'max_input_len': 512,       # Tune: 128-2048. Longer = more context but slower training
    'max_output_len': 128,      # Tune: 32-512. Depends on expected output length
    'temperature': 2.0,         # Tune: 1.0-10.0. Higher = softer probability distributions for KL loss
    'pad_to_max_length': False, # True = consistent batch shapes but more computation
    'truncation': True,         # Whether to truncate inputs longer than max_input_len
}

# Generation Parameters (for evaluation and inference)
GENERATION_PARAMS = {
    'num_beams': 5,             # Tune: 1-10. Higher = better quality but slower. 1 = greedy decoding
    'do_sample': False,         # True = sampling, False = deterministic generation
    'temperature_gen': 1.0,     # Tune: 0.1-2.0. Only used if do_sample=True. Higher = more random
    'top_p': 0.9,              # Tune: 0.1-1.0. Nucleus sampling threshold. Lower = more focused
    'top_k': 50,               # Tune: 1-100. Top-k sampling. Lower = more focused
    'repetition_penalty': 1.0,  # Tune: 1.0-1.5. Penalize repetition. Higher = less repetitive
    'length_penalty': 1.0,      # Tune: 0.5-2.0. Encourage longer/shorter sequences
    'max_new_tokens': 128,      # Maximum tokens to generate during inference
    'early_stopping': True,     # Stop generation early if EOS token generated
}

# ============================================================================
# LOSS FUNCTION PARAMETERS
# ============================================================================

# Distillation Loss Configuration
DISTILLATION_PARAMS = {
    'alpha': 0.5,               # Tune: 0.0-1.0. Balance between hard (0) and soft (1) targets
    'kl_loss_type': 'forward',  # Options: 'forward', 'reverse', 'symmetric'. Forward KL most common
    'temperature_scaling': True, # Whether to apply temperature scaling to teacher outputs
    'label_smoothing': 0.0,     # Tune: 0.0-0.3. Reduces overconfidence. 0.1 often good starting point
}

# Position-Aware N-gram Similarity (PANS) Parameters
PANS_PARAMS = {
    'pans_temperature': 1.0,    # Tune: 0.5-2.0. Temperature for PANS computation
    'pans_reduction': 'mean',   # Options: 'mean', 'sum', 'none'. How to reduce batch dimension
    'pans_normalize': True,     # Whether to normalize PANS by sequence length
}

# AST Loss Parameters  
AST_PARAMS = {
    'ast_weight_type': 'uniform',  # Options: 'uniform', 'depth_weighted', 'frequency_weighted'
    'ast_max_depth': 10,        # Tune: 5-20. Maximum AST depth to consider
    'ast_ignore_leaf_nodes': False,  # Whether to ignore leaf nodes in AST comparison
    'ast_structural_weight': 0.7,    # Tune: 0.3-0.9. Weight for structural vs content similarity
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

EVALUATION_PARAMS = {
    'eval_batch_size': 8,       # Tune: 4-32. Can be larger than training batch_size for efficiency
    'eval_accumulation_steps': 1,    # Accumulate eval batches before computing metrics
    'metric_for_best_model': 'eval_loss',  # Options: 'eval_loss', 'bleu', 'exact_match', etc.
    'greater_is_better': False, # Whether higher metric values are better
    'load_best_model_at_end': True,     # Load best checkpoint at end of training
    'save_total_limit': 3,      # Maximum number of checkpoints to keep
}

# Metrics Configuration
METRICS_PARAMS = {
    'compute_bleu': True,       # Compute BLEU score during evaluation
    'compute_exact_match': True, # Compute exact match accuracy
    'compute_code_metrics': True, # Compute code-specific metrics (syntax, compilation)
    'bleu_smoothing': True,     # Use smoothing for BLEU calculation
    'case_sensitive': False,    # Whether metrics should be case sensitive
}

# ============================================================================
# DYNAMIC HYPERPARAMETER SCHEDULING
# ============================================================================

# Temperature Decay Configuration
TEMPERATURE_DECAY = {
    'enabled': False,           # Enable temperature decay during training
    'initial_temp': 4.0,        # Tune: 2.0-10.0. Starting temperature
    'final_temp': 1.0,          # Tune: 0.5-2.0. Ending temperature  
    'decay_type': 'linear',     # Options: 'linear', 'exponential', 'cosine'
    'decay_start_epoch': 2,     # Tune: 1-5. When to start temperature decay
}

# Alpha Adaptation (automatic alpha tuning based on performance)
ALPHA_ADAPTATION = {
    'enabled': False,           # Enable adaptive alpha based on validation performance
    'adaptation_patience': 3,   # Epochs to wait before adapting alpha
    'alpha_increase_factor': 1.1,  # Tune: 1.05-1.2. How much to increase alpha
    'alpha_decrease_factor': 0.9,   # Tune: 0.8-0.95. How much to decrease alpha
    'min_alpha': 0.1,          # Minimum allowed alpha value
    'max_alpha': 0.9,          # Maximum allowed alpha value
    'improvement_threshold': 0.001,  # Minimum improvement to avoid adaptation
}

# Progressive Loss Weight Scheduling
PROGRESSIVE_SCHEDULING = {
    'enabled': False,           # Enable progressive introduction of loss components
    'schedule_type': 'sigmoid', # Options: 'linear', 'sigmoid', 'exponential'
    'ce_epochs': [0, 5],       # Epochs when CE loss is active
    'kl_epochs': [1, 5],       # Epochs when KL loss is active  
    'pans_epochs': [2, 5],     # Epochs when PANS is active
    'ast_epochs': [3, 5],      # Epochs when AST loss is active
}

# ============================================================================
# HARDWARE AND PERFORMANCE OPTIMIZATION
# ============================================================================

HARDWARE_PARAMS = {
    'device': 'auto',           # Options: 'auto', 'cpu', 'cuda'. Auto detects best available
    'mixed_precision': 'fp16',  # Options: 'no', 'fp16', 'bf16'. Use fp16 for most GPUs
    'dataloader_num_workers': 0,     # Tune: 0-8. Number of subprocesses for data loading
    'dataloader_pin_memory': True,   # Pin memory for faster GPU transfer
    'gradient_checkpointing': False, # Trade compute for memory - slower but uses less GPU memory
}

# Memory Optimization
MEMORY_PARAMS = {
    'max_memory_mb': None,      # Maximum memory usage in MB (None = no limit)
    'cache_dir': None,          # Directory for caching models and datasets
    'low_cpu_mem_usage': False, # Use less CPU memory when loading models
    'torch_compile': False,     # Enable PyTorch 2.0 compilation (experimental)
}

# Performance Monitoring
PERFORMANCE_PARAMS = {
    'profile_memory': False,    # Enable memory profiling
    'profile_time': False,      # Enable time profiling  
    'log_gpu_memory': False,    # Log GPU memory usage
    'benchmark_mode': False,    # Enable benchmarking mode for consistent timing
}

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

DATA_PARAMS = {
    'shuffle_train': True,      # Shuffle training data each epoch
    'shuffle_eval': False,      # Shuffle evaluation data
    'drop_last_batch': False,   # Drop incomplete final batch
    'prefetch_factor': 2,       # Number of batches to prefetch per worker
    'persistent_workers': False, # Keep workers alive between epochs
}

# Data Augmentation
AUGMENTATION_PARAMS = {
    'code_augmentation': False, # Enable code-specific augmentation
    'variable_renaming': False, # Randomly rename variables
    'comment_removal': False,   # Randomly remove comments
    'whitespace_variation': False, # Vary whitespace formatting
    'augmentation_prob': 0.1,   # Tune: 0.0-0.5. Probability of applying augmentation
}

# ============================================================================
# REGULARIZATION PARAMETERS  
# ============================================================================

REGULARIZATION_PARAMS = {
    'dropout_rate': 0.1,        # Tune: 0.0-0.5. Dropout probability in model layers
    'attention_dropout': 0.1,   # Tune: 0.0-0.3. Dropout in attention layers
    'hidden_dropout': 0.1,      # Tune: 0.0-0.3. Dropout in hidden layers
    'layer_norm_eps': 1e-12,    # Layer normalization epsilon
    'initializer_range': 0.02,  # Weight initialization range
}

# Early Stopping
EARLY_STOPPING_PARAMS = {
    'enabled': False,           # Enable early stopping
    'patience': 5,              # Tune: 3-10. Epochs to wait for improvement
    'min_delta': 0.001,         # Tune: 0.0001-0.01. Minimum improvement threshold
    'restore_best_weights': True, # Restore best weights when stopping early
}

# ============================================================================
# DEBUGGING AND LOGGING PARAMETERS
# ============================================================================

DEBUG_PARAMS = {
    'debug_mode': False,        # Enable debug mode with extra logging
    'log_level': 'INFO',        # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_predictions': False,   # Log model predictions during training
    'save_attention_weights': False, # Save attention weights for analysis
    'track_gradients': False,   # Track gradient norms and distributions
}

# Experiment Tracking
EXPERIMENT_PARAMS = {
    'experiment_name': None,    # Name for experiment tracking
    'tags': [],                 # Tags for organizing experiments
    'notes': '',                # Additional notes about the experiment
    'save_code_snapshot': False, # Save code state with experiment
}

# ============================================================================
# PARAMETER TUNING GUIDELINES
# ============================================================================

TUNING_GUIDELINES = {
    'quick_start': {
        'description': 'Fast training with basic settings',
        'suggested_changes': {
            'batch_size': 8,
            'learning_rate': 1e-4,
            'epochs': 3,
            'warmup_steps': 100
        }
    },
    'high_quality': {
        'description': 'Longer training for best results',
        'suggested_changes': {
            'batch_size': 4,
            'learning_rate': 5e-5,
            'epochs': 10,
            'warmup_steps': 1000,
            'eval_strategy': 'steps',
            'eval_steps': 500
        }
    },
    'memory_constrained': {
        'description': 'Settings for limited GPU memory',
        'suggested_changes': {
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'gradient_checkpointing': True,
            'max_input_len': 256,
            'mixed_precision': 'fp16'
        }
    },
    'code_quality_focused': {
        'description': 'Emphasize code quality metrics',
        'suggested_changes': {
            'loss_function': 'multi_component',
            'scheduling_preset': 'code_focused',
            'compute_code_metrics': True,
            'pans_weight': 0.2,
            'ast_weight': 0.15
        }
    }
}

# ============================================================================
# TOKEN-SPECIFIC WEIGHTING PARAMETERS (Task 4 - PRD v1)
# ============================================================================

# Critical token weighting for enhanced assertion generation
DEFAULT_CRITICAL_TOKEN_WEIGHT = 2.0  # Weight multiplier for critical assertion tokens
DEFAULT_ENABLE_TOKEN_WEIGHTING = False  # Enable token-specific weighting by default

# Token weighting configuration
TOKEN_WEIGHTING_PARAMS = {
    'critical_token_weight': DEFAULT_CRITICAL_TOKEN_WEIGHT,  # Tune: 1.5-4.0. Higher = more focus on critical tokens
    'enable_token_weighting': DEFAULT_ENABLE_TOKEN_WEIGHTING,  # Whether to enable token-specific weighting
    'cache_token_mappings': True,  # Cache token-to-vocab mappings for performance
    'apply_to_loss_functions': ['ce', 'focal'],  # Which loss functions use token weighting
}
