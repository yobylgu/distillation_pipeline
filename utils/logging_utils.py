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

class DistillationLogger:
    """
    Comprehensive logging system for knowledge distillation training.
    Tracks loss components, hyperparameters, and provides real-time monitoring.
    """
    
    def __init__(self, output_dir, loss_components=None):
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
        
        # Initialize CSV log file with dynamic columns
        self.csv_file = os.path.join(output_dir, 'training_metrics.csv')
        self._initialize_csv_header()
    
    def _initialize_csv_header(self):
        """Initialize CSV header with dynamic loss components."""
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
