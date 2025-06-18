#!/usr/bin/env python3
"""
Command Utilities for Knowledge Distillation Pipeline

This module provides utilities for formatting and logging training commands,
making it easy to track and reproduce experimental configurations.
"""

def format_command_from_args(args):
    """
    Format the command line arguments back into a command string for logging.
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        str: Formatted command string with proper line continuations
    """
    command_parts = ["python knowledge_distillation.py"]
    
    # Add all arguments in a consistent order
    command_parts.extend([
        f"    --train_data_path {args.train_data_path}",
        f"    --val_data_path {args.val_data_path}",
        f"    --max_train_samples {args.max_train_samples}",
        f"    --max_val_samples {args.max_val_samples}",
        f"    --batch_size {args.batch_size}",
        f"    --epochs {args.epochs}",
        f"    --gradient_accumulation_steps {args.gradient_accumulation_steps}",
        f"    --lr {args.lr}",
        f"    --warmup_steps {args.warmup_steps}",
        f"    --weight_decay {args.weight_decay}",
        f"    --alpha {args.alpha}",
        f"    --temperature {args.temperature}",
        f"    --output_dir {args.output_dir}",
        f"    --loss_function {args.loss_function}",
        f"    --loss_components {' '.join(args.loss_components)}",
        f"    --dropout_rate {args.dropout_rate}",
        f"    --max_input_len {args.max_input_len}",
        f"    --max_output_len {args.max_output_len}",
        f"    --model_name {args.model_name}",
        f"    --seed {args.seed}",
    ])
    
    # Add boolean flags
    if args.enable_dynamic_weighting:
        command_parts.append("    --enable_dynamic_weighting")
    if args.use_enhanced_metrics:
        command_parts.append("    --use_enhanced_metrics")
    
    # Add optional arguments if they differ from defaults
    if args.validation_frequency != 1:
        command_parts.append(f"    --validation_frequency {args.validation_frequency}")
    
    if args.loss_weights:
        command_parts.append(f"    --loss_weights {' '.join(map(str, args.loss_weights))}")
    
    return " \\\n".join(command_parts)


def log_training_command(logger, args, phase="START"):
    """
    Log the training command with proper formatting.
    
    Args:
        logger: DistillationLogger instance
        args: Parsed command line arguments
        phase: str, either "START" or "END" to indicate logging phase
    """
    command_log = format_command_from_args(args)
    
    if phase == "START":
        title = "TRAINING COMMAND USED:"
    elif phase == "END":
        title = "TRAINING COMPLETED - COMMAND USED:"
    else:
        title = "TRAINING COMMAND:"
    
    logger.logger.info("=" * 80)
    logger.logger.info(title)
    logger.logger.info("=" * 80)
    logger.logger.info(command_log)
    logger.logger.info("=" * 80)


def save_training_config_to_json(args, output_dir, filename="training_config.json"):
    """
    Save all training configuration parameters to a JSON file for complete reproducibility.
    This includes both specified arguments and their default values.
    
    Args:
        args: Parsed command line arguments from argparse
        output_dir: str, output directory path
        filename: str, name of the JSON file to save the configuration
        
    Returns:
        str: Path to the saved JSON configuration file
    """
    import os
    import json
    from datetime import datetime
    
    # Create comprehensive configuration dictionary
    config = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "knowledge_distillation.py",
            "config_format_version": "1.0"
        },
        "data": {
            "train_data_path": args.train_data_path,
            "val_data_path": args.val_data_path,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples
        },
        "model": {
            "model_name": args.model_name,
            "max_input_len": args.max_input_len,
            "max_output_len": args.max_output_len,
            "dropout_rate": args.dropout_rate
        },
        "training": {
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "validation_frequency": args.validation_frequency,
            "seed": args.seed
        },
        "distillation": {
            "alpha": args.alpha,
            "temperature": args.temperature,
            "loss_function": args.loss_function,
            "loss_components": args.loss_components,
            "loss_weights": args.loss_weights,
            "enable_dynamic_weighting": args.enable_dynamic_weighting
        },
        "evaluation": {
            "use_enhanced_metrics": args.use_enhanced_metrics
        },
        "command_reproduction": {
            "command": format_command_from_args(args),
            "note": "This command can be used to reproduce this exact training run"
        }
    }
    
    config_file_path = os.path.join(output_dir, filename)
    
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config_file_path


def save_command_to_file(args, output_dir, filename="training_command.txt"):
    """
    DEPRECATED: Use save_training_config_to_json instead.
    
    Save the training command to a separate file for easy access.
    
    Args:
        args: Parsed command line arguments
        output_dir: str, output directory path
        filename: str, name of the file to save the command
    """
    import os
    
    command_log = format_command_from_args(args)
    command_file_path = os.path.join(output_dir, filename)
    
    with open(command_file_path, 'w') as f:
        f.write("# Training Command Used\n")
        f.write("# This command can be used to reproduce this training run\n\n")
        f.write(command_log)
        f.write("\n")
    
    return command_file_path
