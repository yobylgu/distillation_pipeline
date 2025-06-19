#!/usr/bin/env python3
"""
Knowledge Distillation Pipeline for Java Test Assertion Generation

This script implements a comprehensive knowledge distillation framework that trains a smaller 
"student" model to mimic the behavior of a larger pre-trained "teacher" model for generating 
Java unit test assertions. The pipeline supports multiple loss functions, dynamic weight 
scheduling, and advanced evaluation metrics.

Key Features:
- Multi-component loss functions (CE, KL, PANS, AST-based)
- Dynamic weight scheduling during training
- Comprehensive evaluation with code-specific metrics
- Memory-efficient teacher logit compression/decompression
- Modular architecture with extensive configuration options

Example Usage:

# Default Trident loss training (recommended)
python knowledge_distillation.py \
    --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
    --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
    --max_train_samples 200 \
    --max_val_samples 100 \
    --batch_size 4 \
    --epochs 5 \
    --gradient_accumulation_steps 4 \
    --lr 5e-5 \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --alpha 0.7 \
    --output_dir results/test \
    --loss_function multi_component \
    --loss_components focal jsd semantic \
    --enable_dynamic_weighting \
    --seed 42 \
    --num_workers 0 \
    --dropout_rate 0.1 \
    --use_enhanced_metrics \
    --validation_frequency 10 \
    --max_input_len 512 \
    --max_output_len 128 \
    --model_name Salesforce/codet5p-220m

    python knowledge_distillation.py \
      --train_data_path data/codet5p-focal-methods/distillation_data_training.jsonl \
      --val_data_path data/codet5p-focal-methods/distillation_data_validation.jsonl \
      --max_train_samples 5000 \
      --max_val_samples 1000 \
      --batch_size 4 \
      --epochs 10 \
      --gradient_accumulation_steps 4 \
      --lr 5e-5 \
      --warmup_steps 1000 \
      --weight_decay 0.01 \
      --alpha 0.5 \
      --temperature 4 \
      --output_dir results/test_training \
      --loss_function multi_component \
      --loss_components focal jsd semantic \
      --enable_dynamic_weighting \
      --dropout_rate 0.1 \
      --early_stopping_patience 15 \
      --validation_frequency 2 \
      --use_enhanced_metrics \
      --num_workers 4 \
      --sampling_seed 42 \
      --enable_epoch_sampling
For more configuration options, see config/defaults.py or run with --help
"""
import os
import argparse
import torch
import json
import numpy as np
import random
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
import shutil
from torch.nn import Dropout
from torch.cuda.amp import GradScaler, autocast

# Import our modular components
from data import AssertionDataset, EpochSamplingDataset, optimized_collate_fn
from models import (
    optimized_distillation_loss_with_logging,
    enhanced_distillation_loss,
    ast_enhanced_loss,
    MultiComponentLoss
)
from evaluation import fast_evaluate
from utils.logging_utils import DistillationLogger
from utils.training_utils import get_dynamic_hyperparams, setup_loss_function
from utils.device_utils import setup_device
from utils.command_utils import log_training_command, save_training_config_to_json
from config import *
# NEW: Import token weighting defaults (Task 4.2)
from config.defaults import DEFAULT_CRITICAL_TOKEN_WEIGHT, DEFAULT_RUNNING_AVG_MOMENTUM, DEFAULT_LOSS_NORM_WARMUP_STEPS


def set_seed(seed):
    """
    Set all random seeds for reproducible training.
    
    Args:
        seed (int): Random seed value
    """
    print(f"Setting all random seeds to: {seed}")
    
    # Python random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch CPU random number generator
    torch.manual_seed(seed)
    
    # PyTorch GPU random number generators (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Additional CUDA settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Note: These settings may reduce performance but ensure reproducibility
        print("CUDA deterministic mode enabled for reproducibility")
    
    # Set a fixed seed for data loading workers
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("All random number generators have been seeded for reproducibility")


def check_tensor_validity(tensor, name, logger, step=None):
    """Checks for NaN or Inf in a tensor."""
    if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
        logger.logger.error(f"Invalid tensor detected in '{name}' at step {step}: contains NaN or Inf.")
        return False
    return True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=DEFAULT_EARLY_STOPPING_PATIENCE, min_delta=DEFAULT_EARLY_STOPPING_MIN_DELTA, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.logger:
                self.logger.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decreases."""
        if self.logger:
            self.logger.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        best_model_path = os.path.join(path, 'best_model')
        model.save_pretrained(best_model_path)
        
        tokenizer_path = os.path.join(path, 'best_model')
        if hasattr(model, 'tokenizer'):
             model.tokenizer.save_pretrained(tokenizer_path)

        self.val_loss_min = val_loss


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Test Assertion Generation")
    
    # Data arguments
    parser.add_argument('--train_data_path', required=True, help='Path to training data')
    parser.add_argument('--val_data_path', required=True, help='Path to validation data')
    parser.add_argument('--max_train_samples', type=int, default=DEFAULT_MAX_TRAIN_SAMPLES, help='Maximum training samples')
    parser.add_argument('--max_val_samples', type=int, default=DEFAULT_MAX_VAL_SAMPLES, help='Maximum validation samples')
    
    # Model arguments
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, help='Model name or path')
    parser.add_argument('--max_input_len', type=int, default=DEFAULT_MAX_INPUT_LEN, help='Maximum input length')
    parser.add_argument('--max_output_len', type=int, default=DEFAULT_MAX_OUTPUT_LEN, help='Maximum output length')
    
    # Training arguments
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS, help='Warmup steps for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Alpha for CE vs KL loss')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help='Temperature for distillation')
    
    # Loss function arguments
    parser.add_argument('--loss_function', choices=LOSS_FUNCTION_CHOICES, default=DEFAULT_LOSS_FUNCTION, help='Loss function to use')
    parser.add_argument('--loss_components', nargs='+', default=DEFAULT_LOSS_COMPONENTS, choices=LOSS_COMPONENT_CHOICES, help='Loss components for multi_component loss')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=None, help='Weights for loss components')
    
    # MODIFICATION: Arguments from v2 for regularization and dynamic weighting
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('--enable_dynamic_weighting', action='store_true', help='Enable dynamic weight scheduling for multi-component loss')
    
    # NEW: Loss normalization arguments  
    parser.add_argument('--disable_loss_normalization', action='store_true', help='Disable running average normalization for loss components (enabled by default)')
    parser.add_argument('--loss_norm_momentum', type=float, default=DEFAULT_RUNNING_AVG_MOMENTUM, help='Momentum for running average loss normalization (0.9-0.999)')
    parser.add_argument('--loss_norm_warmup_steps', type=int, default=DEFAULT_LOSS_NORM_WARMUP_STEPS, help='Warmup steps before applying loss normalization')
    
    # NEW: Token-specific weighting arguments (Task 4.2)
    parser.add_argument('--enable_token_weighting', action='store_true', help='Enable token-specific weighting for critical assertion tokens')
    parser.add_argument('--critical_token_weight', type=float, default=DEFAULT_CRITICAL_TOKEN_WEIGHT, help='Weight multiplier for critical tokens (default 2.0)')
    
    # Hardware arguments
    parser.add_argument('--device', default=HARDWARE_PARAMS['device'], choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'], 
                       help='Device to use for training (auto, cpu, cuda, or specific GPU like cuda:0)')
    parser.add_argument('--fp16', action='store_true', help='Enable automatic mixed precision (FP16) training')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='Number of data loading workers (default: 4)')
    
    # Other arguments
    parser.add_argument('--use_enhanced_metrics', action='store_true', help='Use enhanced assertion evaluation metrics')
    parser.add_argument('--validation_frequency', type=int, default=DEFAULT_VALIDATION_FREQUENCY, help='Validate every N epochs (set to 0 to disable validation and early stopping)')
    parser.add_argument('--early_stopping_patience', type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA, help='Minimum improvement required to reset early stopping counter')
    parser.add_argument('--enable_epoch_sampling', action='store_true', help='Enable memory-efficient random sampling per epoch')
    parser.add_argument('--sampling_seed', type=int, default=42, help='Random seed for epoch sampling (default: 42)')
    
    # Reproducibility arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible training (default: 42)')

    return parser.parse_args()


# MODIFICATION: Added from v2 to apply dropout
def add_dropout_to_model(model, dropout_rate=0.1):
    """Recursively adds dropout to all applicable layers of a model."""
    for name, module in model.named_children():
        if isinstance(module, Dropout):
            module.p = dropout_rate
        elif hasattr(module, 'dropout') and isinstance(module.dropout, Dropout):
            module.dropout.p = dropout_rate
        
        if len(list(module.children())) > 0:
            add_dropout_to_model(module, dropout_rate)
    return model


# MODIFICATION: Updated to accept and apply dropout_rate
def setup_model_and_tokenizer(model_name, device, dropout_rate):
    """Setup model and tokenizer."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Apply dropout if specified
    if dropout_rate > 0:
        print(f"Applying dropout with rate: {dropout_rate}")
        model = add_dropout_to_model(model, dropout_rate)
        
    model = model.to(device)
    model.tokenizer = tokenizer

    return model, tokenizer


def setup_datasets(args, tokenizer):
    """Setup training and validation datasets."""
    if args.enable_epoch_sampling:
        print("Using EpochSamplingDataset for memory-efficient training")
        train_ds = EpochSamplingDataset(
            args.train_data_path, tokenizer, 
            max_input_len=args.max_input_len, max_output_len=args.max_output_len, 
            max_samples=args.max_train_samples, seed=args.sampling_seed
        )
    else:
        print("Using standard AssertionDataset")
        train_ds = AssertionDataset(
            args.train_data_path, tokenizer, 
            max_input_len=args.max_input_len, max_output_len=args.max_output_len, max_samples=args.max_train_samples
        )
    
    # Validation dataset always uses standard loading (smaller, needs consistency)
    val_ds = AssertionDataset(
        args.val_data_path, tokenizer,
        max_input_len=args.max_input_len, max_output_len=args.max_output_len, max_samples=args.max_val_samples
    )
    
    # For epoch sampling, we need to check total dataset size instead of loaded samples
    if args.enable_epoch_sampling:
        if train_ds.total_dataset_size == 0:
            raise ValueError("Training data is empty. Please check the path and content.")
    else:
        if not train_ds:
            raise ValueError("Training data is empty. Please check the path and content.")
    
    return train_ds, val_ds


def worker_init_fn(worker_id):
    """Initialize data loading workers with deterministic seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_deterministic_generator(seed):
    """Create a deterministic torch Generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_deterministic_dataloader(dataset, args, shuffle=False, seed=None):
    """Create a fully deterministic DataLoader."""
    if seed is None:
        seed = args.seed
    
    generator = create_deterministic_generator(seed) if shuffle else None
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=optimized_collate_fn
    )


def setup_training(args, train_ds, model):
    """Setup data loaders, optimizer, and scheduler."""
    # For epoch sampling, we'll create the train_loader dynamically each epoch
    if args.enable_epoch_sampling:
        # Create a dummy loader to estimate total steps (will be recreated each epoch)
        # Assume max_samples for step calculation
        estimated_steps_per_epoch = max(1, args.max_train_samples // args.batch_size)
        total_steps = estimated_steps_per_epoch // args.gradient_accumulation_steps * args.epochs
        train_loader = None  # Will be created each epoch
    else:
        train_loader = create_deterministic_dataloader(
            train_ds, args, shuffle=True, seed=args.seed
        )
        total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    return train_loader, optimizer, scheduler


# MODIFICATION: Added dynamic weight logic from v2
def train_epoch(model, train_loader, optimizer, scheduler, logger, loss_fn, multi_loss, args, epoch, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # Initialize epoch loss tracking - use actual components from multi_loss if available
    if args.loss_function == 'multi_component' and multi_loss:
        loss_components_to_track = multi_loss.components + ['total']
    else:
        loss_components_to_track = ['ce', 'kl', 'total']
    
    epoch_losses = {comp: [] for comp in loss_components_to_track}
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')

    for step, batch in enumerate(progress_bar):
        inp = batch['input_ids'].to(device)
        att = batch['attention_mask'].to(device)
        lbl = batch['labels'].to(device)
        teacher_logits = batch['teacher_logits'].to(device)

        # Use autocast for mixed precision if scaler is provided
        use_amp = scaler is not None
        
        with autocast() if use_amp else torch.no_grad() if False else torch.enable_grad():
            outputs = model(input_ids=inp, attention_mask=att, labels=lbl)
            
            if not check_tensor_validity(outputs.logits, "student_logits", logger, step):
                continue

            temperature, alpha = get_dynamic_hyperparams(epoch, args.epochs, logger.get_loss_history())
            
            loss_components = {}
            if args.loss_function == 'multi_component' and multi_loss:
                # Dynamic weight update logic from v2
                if args.enable_dynamic_weighting and hasattr(multi_loss, 'update_weights'):
                     old_weights = multi_loss.get_current_weights() if hasattr(multi_loss, 'get_current_weights') else None
                     multi_loss.update_weights(epoch, args.epochs)
                     new_weights = multi_loss.get_current_weights() if hasattr(multi_loss, 'get_current_weights') else None
                     if old_weights and new_weights and old_weights != new_weights:
                        logger.logger.info(f"Weight update at epoch {epoch+1}: {new_weights}")

                loss, loss_components = multi_loss.compute(outputs.logits, teacher_logits, lbl, T=temperature, alpha=alpha)
            elif args.loss_function == 'traditional':
                 loss, loss_components = optimized_distillation_loss_with_logging(outputs.logits, teacher_logits, lbl, T=temperature, alpha=alpha)
            else: 
                loss = outputs.loss
                loss_components['total'] = loss.item()
                loss_components['ce'] = loss.item()

            if not check_tensor_validity(loss, "loss", logger, step):
                continue
                
            loss = loss / args.gradient_accumulation_steps

        # Use scaled backward pass for AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # NEW: Log detailed step metrics including gradient norms (Task 3.1)
        mini_batch_idx = step % args.gradient_accumulation_steps
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        logger.log_step_detailed(
            epoch+1, step, mini_batch_idx, loss_components, 
            {'temperature': temperature, 'alpha': alpha},
            optimizer, model, effective_batch_size
        )

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if use_amp:
                # Unscale gradients before clipping for AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=DEFAULT_MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=DEFAULT_MAX_GRAD_NORM)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Store loss components for epoch summary (like v1/v2)
        for comp in loss_components_to_track:
            if comp in loss_components:
                epoch_losses[comp].append(loss_components[comp])
        
        progress_bar.set_postfix({"loss": f"{loss_components.get('total', 0):.4f}"})
        logger.log_step(epoch+1, step, loss_components, {
            'lr': scheduler.get_last_lr()[0],
            'temperature': temperature,
            'alpha': alpha
        }, optimizer)
        
    avg_loss = total_loss / len(train_loader)
    
    # Pass accumulated epoch losses to logger like v1/v2
    logger.log_epoch(epoch+1, epoch_losses)
    return avg_loss


def validate_epoch(model, val_loader, device, tokenizer, loss_fn, multi_loss, args, epoch, logger):
    """Validate model for one epoch."""
    model.eval()
    total_val_loss = 0
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}')
    with torch.no_grad():
        for batch in progress_bar:
            inp = batch['input_ids'].to(device)
            att = batch['attention_mask'].to(device)
            lbl = batch['labels'].to(device)
            
            outputs = model(input_ids=inp, attention_mask=att, labels=lbl)
            loss = outputs.loss
            
            if check_tensor_validity(loss, "val_loss", logger):
                total_val_loss += loss.item()
                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = total_val_loss / len(val_loader) if val_loader else 0
    return avg_val_loss

# MODIFICATION: Updated to pass dropout_rate to setup function
def main():
    """Main training and evaluation function."""
    args = parse_arguments()
    
    # Set all random seeds for reproducibility
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_version_str = args.model_name.replace("/", "-")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{model_version_str}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    device = setup_device(args.device)
    
    # Log training configuration summary
    print(f"Training Configuration Summary:")
    print(f"  Device: {device}")
    print(f"  FP16: {args.fp16} ({'will be enabled' if torch.cuda.is_available() and device.type == 'cuda' and args.fp16 else 'disabled/not supported'})")
    print(f"  DataLoader workers: {args.num_workers}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    
    # Pass the dropout rate to the setup function
    model, tokenizer = setup_model_and_tokenizer(args.model_name, device, args.dropout_rate)

    train_ds, val_ds = setup_datasets(args, tokenizer)
    train_loader, optimizer, scheduler = setup_training(args, train_ds, model)

    val_loader = create_deterministic_dataloader(
        val_ds, args, shuffle=False, seed=args.seed
    )

    # Load sentence transformer model if semantic loss component is used
    sentence_transformer_model = None
    if args.loss_function == 'multi_component' and 'semantic' in args.loss_components:
        print(f"Semantic loss component detected in: {args.loss_components}")
        try:
            print("Attempting to import sentence_transformers...")
            from sentence_transformers import SentenceTransformer
            print("✓ sentence_transformers imported successfully")
            print("Loading sentence transformer model 'embaas/codesearchnet-minilm-l6' for semantic loss...")
            sentence_transformer_model = SentenceTransformer('embaas/codesearchnet-minilm-l6')
            print("✓ Sentence transformer model loaded successfully")
            print(f"Model device: {sentence_transformer_model.device}")
        except ImportError as e:
            print(f"✗ ImportError: sentence-transformers not installed: {e}")
            print("  Install with: pip install sentence-transformers")
            print("  Removing 'semantic' from loss components to prevent training failure")
            args.loss_components = [comp for comp in args.loss_components if comp != 'semantic']
            print(f"  Updated components: {args.loss_components}")
        except Exception as e:
            print(f"✗ Failed to load sentence transformer model: {e}")
            print(f"  Error type: {type(e).__name__}")
            print("  This could be due to:")
            print("    - Network connectivity issues (model download)")
            print("    - Insufficient memory")
            print("    - PyTorch compatibility issues")
            print("  Removing 'semantic' from loss components to prevent training failure")
            args.loss_components = [comp for comp in args.loss_components if comp != 'semantic']
            print(f"  Updated components: {args.loss_components}")
    else:
        if args.loss_function == 'multi_component':
            print(f"No semantic loss component in: {args.loss_components}")
        else:
            print(f"Using {args.loss_function} loss (not multi_component), semantic loss not applicable")

    loss_fn, multi_loss = setup_loss_function(args, tokenizer, sentence_transformer_model)
    
    # Determine loss components dynamically based on the configured loss function
    if args.loss_function == 'multi_component' and multi_loss:
        # Get actual components from the multi_loss instance plus 'total'
        loss_components = multi_loss.components + ['total']
    else:
        # Default components for traditional loss functions
        loss_components = ['ce', 'kl', 'total']
    
    logger = DistillationLogger(output_dir, loss_components)
    
    # Debug logging status for Colab troubleshooting
    logger.debug_logging_status()
    
    # Initialize GradScaler for AMP if FP16 is enabled and CUDA is available
    scaler = None
    if args.fp16:
        if torch.cuda.is_available() and device.type == 'cuda':
            scaler = GradScaler()
            logger.logger.info(f"Automatic Mixed Precision (FP16) enabled on {device}")
        else:
            logger.logger.warning(f"FP16 requested but not supported on device {device}, falling back to FP32")
    else:
        logger.logger.info(f"Training in FP32 mode on device {device}")
    
    # Log the training command and save configuration to JSON
    log_training_command(logger, args, phase="START")
    config_file_path = save_training_config_to_json(args, output_dir)
    logger.logger.info(f"Complete training configuration saved to: {config_file_path}")
    
    # Only initialize early stopping if validation is enabled
    early_stopping = None
    if args.validation_frequency > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta, logger=logger)
        logger.logger.info(f"Early stopping enabled with validation every {args.validation_frequency} epoch(s), patience={args.early_stopping_patience}")
    else:
        logger.logger.info("Validation disabled (validation_frequency=0), early stopping will not be used")

    for epoch in range(args.epochs):
        # Handle epoch sampling: resample data and recreate data loader
        if args.enable_epoch_sampling:
            logger.logger.info(f"=== Epoch {epoch+1}: Resampling training data ===")
            train_ds.resample_for_epoch(epoch)
            
            # Recreate the training loader with new data (deterministic)
            train_loader = create_deterministic_dataloader(
                train_ds, args, shuffle=True, seed=args.seed + epoch
            )
            
            # Update scheduler total steps if this is the first epoch (since data size might differ)
            if epoch == 0:
                new_total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
                logger.logger.info(f"Updated total training steps to {new_total_steps} (based on actual data size)")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, logger, loss_fn, multi_loss, args, epoch, device, scaler)
        
        # Clear epoch data after training to free memory
        if args.enable_epoch_sampling:
            logger.logger.info(f"Clearing epoch {epoch+1} data to free memory")
            train_ds.clear_epoch_data()
        
        # Only validate if validation_frequency > 0 and it's the right epoch or the final epoch
        should_validate = (args.validation_frequency > 0 and 
                          ((epoch + 1) % args.validation_frequency == 0 or epoch == args.epochs - 1))
        
        if should_validate:
            val_loss = validate_epoch(model, val_loader, device, tokenizer, loss_fn, multi_loss, args, epoch, logger)
            logger.logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Only use early stopping if it's enabled
            if early_stopping:
                early_stopping(val_loss, model, output_dir)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        else:
            logger.logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f} (validation skipped)')        
            
    final_model_path = os.path.join(output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.logger.info(f"Final model state saved to {final_model_path}")

    # Final evaluation - always use final model
    logger.logger.info("Using final model for final evaluation.")
    evaluation_model = model
    model_type = "Final Model"
    
    # Only run final evaluation if validation was enabled at least once
    if args.validation_frequency > 0:
        final_metrics = fast_evaluate(
            evaluation_model, tokenizer, val_ds, output_dir, device,
            use_enhanced_metrics=args.use_enhanced_metrics
        )
        logger.logger.info(f"--- Final Metrics (from {model_type}) ---")
        # Format metrics with 3 decimal precision for display
        formatted_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, float):
                formatted_metrics[key] = round(value, 3)
            else:
                formatted_metrics[key] = value
        logger.logger.info(json.dumps(formatted_metrics, indent=2))
    else:
        logger.logger.info("Skipping final evaluation since validation was disabled (validation_frequency=0)")

    # Final cleanup for epoch sampling
    if args.enable_epoch_sampling:
        logger.logger.info("Final cleanup: clearing any remaining epoch data")
        train_ds.clear_epoch_data()

    # Log the command again at the end for easy reference
    log_training_command(logger, args, phase="END")

    logger.save_final_report()

if __name__ == '__main__':
    main()