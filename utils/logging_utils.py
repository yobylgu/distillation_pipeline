"""
Logging utilities for knowledge distillation training.
"""
import os
import csv
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# TensorBoard support (Task 3.2)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class DistillationLogger:
    """
    Comprehensive logging system for knowledge distillation training.
    Tracks loss components, hyperparameters, and provides real-time monitoring.
    """
    
    def __init__(self, output_dir, loss_components=None, enable_tensorboard=True):
        self.output_dir = output_dir
        # Support for additional loss components
        if loss_components is None:
            loss_components = ['ce', 'kl', 'total']
        self.loss_components = loss_components
        self.loss_history = {comp: [] for comp in loss_components}
        
        self.hyperparams_history = []
        self.step_logs = []
        self.start_time = time.time()
        
        # Setup logging
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'distillation_log.txt')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # NEW: TensorBoard support (Task 3.2)
        self.tensorboard_writer = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tensorboard_dir = os.path.join(output_dir, 'tensorboard')
                self.tensorboard_writer = SummaryWriter(tensorboard_dir)
                self.logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
        elif enable_tensorboard and not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard requested but not available. Install tensorboard package.")
        
        # Initialize CSV log file with dynamic columns
        self.csv_file = os.path.join(output_dir, 'training_metrics.csv')
        # NEW: Initialize per-step CSV file for detailed logging (Task 3.1)
        self.step_metrics_file = os.path.join(output_dir, 'step_metrics.csv')
        self._initialize_csv_headers()
    
    def _initialize_csv_headers(self):
        """Initialize CSV headers for both training metrics and detailed step metrics."""
        # Initialize main training metrics CSV (epoch-level)
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Base columns
            columns = ['timestamp', 'epoch', 'step']
            # Add loss component columns dynamically
            for comp in self.loss_components:
                columns.append(f'{comp}_loss')
            # Add hyperparameter columns
            columns.extend(['temperature', 'alpha', 'learning_rate', 'elapsed_time'])
            writer.writerow(columns)
        
        # NEW: Initialize detailed step metrics CSV (Task 3.1)
        with open(self.step_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Enhanced columns including gradient norms
            columns = ['timestamp', 'epoch', 'step', 'mini_batch_idx']
            
            # Loss components (raw and weighted scalars)
            for comp in self.loss_components:
                columns.extend([f'{comp}_loss_raw', f'{comp}_loss_weighted'])
            
            # Gradient norms and training metrics
            columns.extend([
                'grad_norm_total',
                'grad_norm_encoder', 
                'grad_norm_decoder',
                'learning_rate',
                'temperature',
                'alpha',
                'effective_batch_size',
                'memory_usage_mb',
                'elapsed_time_seconds'
            ])
            
            # Component weights for multi-component loss
            for comp in self.loss_components:
                if comp != 'total':
                    columns.append(f'{comp}_weight')
            
            writer.writerow(columns)
    
    def log_step(self, epoch, step, losses, hyperparams, optimizer):
        """
        Log training step information.
        
        Args:
            epoch: Current epoch
            step: Current step
            losses: Dict with 'ce', 'kl', 'total' losses
            hyperparams: Dict with 'temperature', 'alpha'
            optimizer: Optimizer object to get learning rate
        """
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        lr = optimizer.param_groups[0]['lr']
        
        # Store in memory
        step_log = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'losses': losses.copy(),
            'hyperparams': hyperparams.copy(),
            'learning_rate': lr,
            'elapsed_time': elapsed_time
        }
        self.step_logs.append(step_log)
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Build row with all loss components dynamically
            row = [timestamp, epoch, step]
            for comp in self.loss_components:
                row.append(losses.get(comp, 0.0))
            row.extend([
                hyperparams['temperature'], hyperparams['alpha'], lr, elapsed_time
            ])
            writer.writerow(row)
        
        # Console output every 10 steps with enhanced info
        if step % 10 == 0:
            loss_str = f"Loss={losses.get('total', 0):.4f} ("
            loss_parts = []
            for comp in self.loss_components:
                if comp != 'total' and comp in losses and losses[comp] is not None:
                    # Only show non-zero values to avoid cluttering output
                    if losses[comp] == 0:
                        continue
                    loss_parts.append(f"{comp.upper()}={losses[comp]:.4f}")
            loss_str += ", ".join(loss_parts) + ")"
            
            self.logger.info(
                f"Epoch {epoch}, Step {step}: "
                f"{loss_str} "
                f"T={hyperparams['temperature']:.3f}, α={hyperparams['alpha']:.3f}"
            )
        
        # NEW: Log to TensorBoard (Task 3.2)
        if self.tensorboard_writer:
            global_step = (epoch - 1) * 1000 + step  # Approximate global step
            
            # Log loss components
            for comp in self.loss_components:
                if comp in losses and losses[comp] is not None:
                    self.tensorboard_writer.add_scalar(f'Loss/{comp}', losses[comp], global_step)
            
            # Log hyperparameters
            self.tensorboard_writer.add_scalar('Hyperparams/temperature', hyperparams['temperature'], global_step)
            self.tensorboard_writer.add_scalar('Hyperparams/alpha', hyperparams['alpha'], global_step)
            self.tensorboard_writer.add_scalar('Hyperparams/learning_rate', lr, global_step)
            
            # Log metadata if available
            if '_meta' in losses:
                meta = losses['_meta']
                
                # Log component weights
                weights = meta.get('weights', {})
                for comp, weight in weights.items():
                    self.tensorboard_writer.add_scalar(f'Weights/{comp}', weight, global_step)
                
                # Log raw vs weighted scalars
                raw_scalars = meta.get('raw_scalars', {})
                weighted_scalars = meta.get('weighted_scalars', {})
                for comp in raw_scalars:
                    self.tensorboard_writer.add_scalar(f'Loss_Raw/{comp}', raw_scalars[comp], global_step)
                if weighted_scalars:
                    for comp in weighted_scalars:
                        self.tensorboard_writer.add_scalar(f'Loss_Weighted/{comp}', weighted_scalars[comp], global_step)
    
    def log_step_detailed(self, epoch, step, mini_batch_idx, losses, hyperparams, 
                         optimizer, model, effective_batch_size=None):
        """
        NEW: Log detailed step metrics including gradient norms (Task 3.1).
        
        Args:
            epoch: Current epoch
            step: Current step  
            mini_batch_idx: Index within the current step (for gradient accumulation)
            losses: Dict with loss components including metadata
            hyperparams: Dict with 'temperature', 'alpha'
            optimizer: Optimizer object to get learning rate
            model: Model for gradient norm computation
            effective_batch_size: Effective batch size (batch_size * gradient_accumulation_steps)
        """
        import torch
        import psutil
        
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        lr = optimizer.param_groups[0]['lr']
        
        # Compute gradient norms
        total_norm = 0.0
        encoder_norm = 0.0
        decoder_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Categorize gradients
                if 'encoder' in name:
                    encoder_norm += param_norm.item() ** 2
                elif 'decoder' in name:
                    decoder_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        encoder_norm = encoder_norm ** (1. / 2)
        decoder_norm = decoder_norm ** (1. / 2)
        
        # Get memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Extract metadata from losses if available
        metadata = losses.get('_meta', {})
        raw_scalars = metadata.get('raw_scalars', {})
        weighted_scalars = metadata.get('weighted_scalars', {})
        weights = metadata.get('weights', {})
        
        # Write detailed metrics to step CSV
        with open(self.step_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Build row
            row = [timestamp, epoch, step, mini_batch_idx]
            
            # Loss components (raw and weighted)
            for comp in self.loss_components:
                raw_value = raw_scalars.get(comp, losses.get(comp, 0.0))
                weighted_value = weighted_scalars.get(comp, 0.0)
                row.extend([raw_value, weighted_value])
            
            # Gradient norms and training metrics
            row.extend([
                total_norm,
                encoder_norm, 
                decoder_norm,
                lr,
                hyperparams.get('temperature', 0.0),
                hyperparams.get('alpha', 0.0),
                effective_batch_size or 0,
                memory_usage,
                elapsed_time
            ])
            
            # Component weights
            for comp in self.loss_components:
                if comp != 'total':
                    row.append(weights.get(comp, 0.0))
            
            writer.writerow(row)
        
        # NEW: Log detailed metrics to TensorBoard (Task 3.2)
        if self.tensorboard_writer:
            global_step = (epoch - 1) * 1000 + step
            
            # Log gradient norms
            self.tensorboard_writer.add_scalar('Gradients/total_norm', total_norm, global_step)
            self.tensorboard_writer.add_scalar('Gradients/encoder_norm', encoder_norm, global_step)
            self.tensorboard_writer.add_scalar('Gradients/decoder_norm', decoder_norm, global_step)
            
            # Log training metrics
            self.tensorboard_writer.add_scalar('Training/effective_batch_size', effective_batch_size or 0, global_step)
            self.tensorboard_writer.add_scalar('System/memory_usage_mb', memory_usage, global_step)
            self.tensorboard_writer.add_scalar('Training/elapsed_time', elapsed_time, global_step)
            
            # Log mini-batch information
            self.tensorboard_writer.add_scalar('Training/mini_batch_idx', mini_batch_idx, global_step)
    
    def log_epoch(self, epoch, epoch_losses, val_metrics=None):
        """Log end-of-epoch summary with support for multiple loss components."""
        # Update loss history for all configured components
        for comp in self.loss_components:
            if comp in epoch_losses and epoch_losses[comp]:
                self.loss_history[comp].append(np.mean(epoch_losses[comp]))
        
        # Log epoch summary
        self.logger.info(f"Epoch {epoch} Summary:")
        for comp in self.loss_components:
            if comp in self.loss_history and self.loss_history[comp]:
                self.logger.info(f"  Avg {comp.upper()} Loss: {self.loss_history[comp][-1]:.4f}")
        
        if val_metrics:
            self.logger.info(f"  Validation F1: {val_metrics.get('f1', 0):.4f}")
            self.logger.info(f"  Validation BLEU: {val_metrics.get('bleu', 0):.4f}")
            # Log enhanced metrics if available
            if 'ast_validity' in val_metrics:
                self.logger.info(f"  AST Validity: {val_metrics['ast_validity']:.4f}")
            if 'code_quality_score' in val_metrics:
                self.logger.info(f"  Code Quality: {val_metrics['code_quality_score']:.4f}")
        
        # Save epoch summary to JSON
        epoch_summary = {
            'epoch': epoch,
            'avg_losses': {},
            'validation_metrics': val_metrics or {},
            'elapsed_time': time.time() - self.start_time
        }
        
        # Include all available loss components
        for comp in self.loss_history:
            if self.loss_history[comp]:
                epoch_summary['avg_losses'][comp] = self.loss_history[comp][-1]
        
        summary_file = os.path.join(self.output_dir, f'epoch_{epoch}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(epoch_summary, f, indent=2)
        
        # NEW: Log epoch metrics to TensorBoard (Task 3.2)
        if self.tensorboard_writer:
            # Log average epoch losses
            for comp in epoch_summary['avg_losses']:
                self.tensorboard_writer.add_scalar(f'Epoch_Loss/{comp}', epoch_summary['avg_losses'][comp], epoch)
            
            # Log validation metrics if available
            if val_metrics:
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
            
            # Log training time
            self.tensorboard_writer.add_scalar('Training/epoch_time', epoch_summary['elapsed_time'], epoch)
    
    def get_loss_history(self):
        """Return loss history for dynamic scheduling."""
        return self.loss_history
    
    def save_final_report(self):
        """Save comprehensive training report."""
        final_losses = {}
        for comp in self.loss_components:
            if self.loss_history[comp]:
                final_losses[comp] = self.loss_history[comp][-1]
            else:
                final_losses[comp] = 0
        
        report = {
            'training_summary': {
                'total_steps': len(self.step_logs),
                'total_time': time.time() - self.start_time,
                'final_losses': final_losses
            },
            'loss_trends': self.loss_history,
            'hyperparameter_evolution': self.hyperparams_history
        }
        
        report_file = os.path.join(self.output_dir, 'training_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training report saved to {report_file}")
        
        # NEW: Close TensorBoard writer (Task 3.2)
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.logger.info("TensorBoard writer closed")
    
    def close(self):
        """
        NEW: Close logger and TensorBoard writer (Task 3.2).
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
            self.logger.info("TensorBoard writer closed")
