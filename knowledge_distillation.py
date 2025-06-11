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
    --max_train_samples 5000 \
    --max_val_samples 100 \
    --batch_size 4 \
    --epochs 5 \
    --gradient_accumulation_steps 4 \
    --lr 2e-5 \
    --warmup_steps 0 \
    --weight_decay 0.01 \
    --alpha 0.7 \
    --temperature 4.0 \
    --output_dir results/distillation_run \
    --loss_function multi_component \
    --loss_components focal jsd semantic \
    --enable_dynamic_weighting \
    --dropout_rate 0.1 \
    --early_stopping_patience 10
    --use_enhanced_metrics \
    --validation_frequency 10 \
    --max_input_len 512 \
    --max_output_len 128 \
    --model_name Salesforce/codet5p-220m

For more configuration options, see config/defaults.py or run with --help
"""
import os
import argparse
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
import shutil
from torch.nn import Dropout #

# Import our modular components
from data import AssertionDataset, optimized_collate_fn
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
from config.defaults import DEFAULT_CRITICAL_TOKEN_WEIGHT


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
    
    # NEW: Token-specific weighting arguments (Task 4.2)
    parser.add_argument('--enable_token_weighting', action='store_true', help='Enable token-specific weighting for critical assertion tokens')
    parser.add_argument('--critical_token_weight', type=float, default=DEFAULT_CRITICAL_TOKEN_WEIGHT, help='Weight multiplier for critical tokens (default 2.0)')
    
    # Hardware arguments
    parser.add_argument('--device', default=HARDWARE_PARAMS['device'], choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'], 
                       help='Device to use for training (auto, cpu, cuda, or specific GPU like cuda:0)')
    
    # Other arguments
    parser.add_argument('--use_enhanced_metrics', action='store_true', help='Use enhanced assertion evaluation metrics')
    parser.add_argument('--validation_frequency', type=int, default=DEFAULT_VALIDATION_FREQUENCY, help='Validate every N epochs')

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
    train_ds = AssertionDataset(
        args.train_data_path, tokenizer, 
        max_input_len=args.max_input_len, max_output_len=args.max_output_len, max_samples=args.max_train_samples
    )
    val_ds = AssertionDataset(
        args.val_data_path, tokenizer,
        max_input_len=args.max_input_len, max_output_len=args.max_output_len, max_samples=args.max_val_samples
    )
    if not train_ds:
        raise ValueError("Training data is empty. Please check the path and content.")
    return train_ds, val_ds


def setup_training(args, train_ds, model):
    """Setup data loaders, optimizer, and scheduler."""
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=optimized_collate_fn, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    return train_loader, optimizer, scheduler


# MODIFICATION: Added dynamic weight logic from v2
def train_epoch(model, train_loader, optimizer, scheduler, logger, loss_fn, multi_loss, args, epoch, device):
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
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_version_str = args.model_name.replace("/", "-")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{model_version_str}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    device = setup_device(args.device)
    # Pass the dropout rate to the setup function
    model, tokenizer = setup_model_and_tokenizer(args.model_name, device, args.dropout_rate)

    train_ds, val_ds = setup_datasets(args, tokenizer)
    train_loader, optimizer, scheduler = setup_training(args, train_ds, model)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=optimized_collate_fn, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True
    )

    # Load sentence transformer model if semantic loss component is used
    sentence_transformer_model = None
    if args.loss_function == 'multi_component' and 'semantic' in args.loss_components:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence transformer model for semantic loss...")
            sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded successfully")
        except ImportError:
            print("Warning: sentence-transformers not installed. Semantic loss will be disabled.")
            # Remove semantic from components to prevent errors
            args.loss_components = [comp for comp in args.loss_components if comp != 'semantic']
        except Exception as e:
            print(f"Warning: Failed to load sentence transformer model: {e}")
            args.loss_components = [comp for comp in args.loss_components if comp != 'semantic']

    loss_fn, multi_loss = setup_loss_function(args, tokenizer, sentence_transformer_model)
    
    # Determine loss components dynamically based on the configured loss function
    if args.loss_function == 'multi_component' and multi_loss:
        # Get actual components from the multi_loss instance plus 'total'
        loss_components = multi_loss.components + ['total']
    else:
        # Default components for traditional loss functions
        loss_components = ['ce', 'kl', 'total']
    
    logger = DistillationLogger(output_dir, loss_components)
    
    # Log the training command and save configuration to JSON
    log_training_command(logger, args, phase="START")
    config_file_path = save_training_config_to_json(args, output_dir)
    logger.logger.info(f"Complete training configuration saved to: {config_file_path}")
    
    early_stopping = EarlyStopping(patience=DEFAULT_EARLY_STOPPING_PATIENCE, logger=logger)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, logger, loss_fn, multi_loss, args, epoch, device)
        
        val_loss = validate_epoch(model, val_loader, device, tokenizer, loss_fn, multi_loss, args, epoch, logger)
        logger.logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        early_stopping(val_loss, model, output_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    final_model_path = os.path.join(output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.logger.info(f"Final model state saved to {final_model_path}")

    best_model_path = os.path.join(output_dir, 'best_model')
    if os.path.exists(best_model_path):
        logger.logger.info(f"Loading best model from {best_model_path} for final evaluation.")
        best_model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path).to(device)
        
        final_metrics = fast_evaluate(
            best_model, tokenizer, val_ds, output_dir, device,
            use_enhanced_metrics=args.use_enhanced_metrics
        )
        logger.logger.info("--- Final Metrics (from Best Model) ---")
        # Format metrics with 3 decimal precision for display
        formatted_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, float):
                formatted_metrics[key] = round(value, 3)
            else:
                formatted_metrics[key] = value
        logger.logger.info(json.dumps(formatted_metrics, indent=2))

    # Log the command again at the end for easy reference
    log_training_command(logger, args, phase="END")

    logger.save_final_report()

if __name__ == '__main__':
    main()