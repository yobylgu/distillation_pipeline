#!/usr/bin/env python3
"""
Test script for Google Colab logging functionality.
Run this in Colab to diagnose logging issues.
"""

import os
import sys
import tempfile
from utils.logging_utils import DistillationLogger, is_colab, setup_colab_logging

def test_colab_logging():
    """Test logging functionality in Google Colab."""
    
    print("=== COLAB LOGGING TEST ===")
    print(f"Running in Colab: {is_colab()}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Setup colab logging
    setup_colab_logging()
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize logger
        try:
            logger = DistillationLogger(
                output_dir=temp_dir,
                loss_components=['ce', 'kl', 'total'],
                enable_tensorboard=False  # Disable for testing
            )
            print("✓ Logger initialized successfully")
            
            # Debug logging status
            status = logger.debug_logging_status()
            
            # Test logging some sample data
            print("\n=== TESTING LOG OPERATIONS ===")
            
            # Test step logging
            sample_losses = {'ce': 0.5, 'kl': 0.3, 'total': 0.8}
            sample_hyperparams = {'temperature': 4.0, 'alpha': 0.5}
            
            # Mock optimizer for testing
            class MockOptimizer:
                def __init__(self):
                    self.param_groups = [{'lr': 0.0001}]
            
            mock_optimizer = MockOptimizer()
            
            # Log several steps
            for step in range(1, 6):
                logger.log_step(
                    epoch=1,
                    step=step,
                    losses=sample_losses,
                    hyperparams=sample_hyperparams,
                    optimizer=mock_optimizer
                )
                print(f"✓ Step {step} logged")
            
            # Test epoch logging
            epoch_losses = {
                'ce': [0.5, 0.4, 0.3],
                'kl': [0.3, 0.2, 0.1],
                'total': [0.8, 0.6, 0.4]
            }
            
            val_metrics = {
                'f1': 0.85,
                'bleu': 0.75,
                'code_quality_score': 0.80
            }
            
            logger.log_epoch(
                epoch=1,
                epoch_losses=epoch_losses,
                val_metrics=val_metrics
            )
            print("✓ Epoch logged")
            
            # Check if files were created and have content
            print("\n=== CHECKING OUTPUT FILES ===")
            
            log_file = os.path.join(temp_dir, 'distillation_log.txt')
            csv_file = os.path.join(temp_dir, 'training_metrics.csv')
            step_file = os.path.join(temp_dir, 'step_metrics.csv')
            
            files_to_check = [
                ('Log file', log_file),
                ('CSV file', csv_file),
                ('Step metrics file', step_file)
            ]
            
            for name, filepath in files_to_check:
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"✓ {name}: EXISTS ({size} bytes)")
                    
                    # Show first few lines
                    try:
                        with open(filepath, 'r') as f:
                            lines = f.readlines()[:5]
                            print(f"  First {len(lines)} lines:")
                            for i, line in enumerate(lines, 1):
                                print(f"    {i}: {line.strip()}")
                    except Exception as e:
                        print(f"  ✗ Error reading file: {e}")
                else:
                    print(f"✗ {name}: NOT FOUND")
            
            # Save final report
            logger.save_final_report()
            print("✓ Final report saved")
            
            # Close logger
            logger.close()
            print("✓ Logger closed")
            
            print("\n=== TEST COMPLETED ===")
            return True
            
        except Exception as e:
            print(f"✗ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_colab_logging()
    if success:
        print("\n🎉 All tests passed! Logging should work in your training scripts.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
    
    print("\n=== USAGE INSTRUCTIONS FOR COLAB ===")
    print("1. Make sure to run this test script first to verify logging works")
    print("2. In your training script, add debug logging:")
    print("   logger = DistillationLogger(output_dir='./results')")
    print("   logger.debug_logging_status()  # Add this line")
    print("3. Check the console output for any error messages")
    print("4. Files are saved to the output directory - check there manually")
    print("5. If issues persist, try mounting Google Drive for persistent storage")