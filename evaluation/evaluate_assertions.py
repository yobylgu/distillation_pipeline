#!/usr/bin/env python3
"""
Post-hoc evaluation script for knowledge distillation pipeline.

This script analyzes pre-computed prediction files from teacher and student models
in a knowledge distillation pipeline for Java test assertion generation.

This corrected version ensures that the student model's generation parameters
(like max_length) are set appropriately to avoid truncated outputs, leading to
a fair and accurate evaluation.

Results are saved to evaluation/post_hoc_evaluation/ directory.

Usage:
    # Mode 1: Evaluate teacher model only
    python evaluation/evaluate_assertions.py \
        --teacher_data data/codet5p-focal-methods/distillation_data_validation.jsonl

    # Mode 2: Evaluate both teacher and student with comparison
    python evaluation/evaluate_assertions.py \
        --teacher_data data/codet5p-focal-methods/distillation_data_validation.jsonl \
        --student_model_path results/test_2025-06-06_17-11-03_Salesforce-codet5p-220m/final_model \
        --student_limit 100
"""

import json
import argparse
import math
import lz4.frame
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import difflib
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# Import existing metrics from evaluators.py
from evaluators import (
    compute_codebleu,
    compute_bleu,
    compute_pans_score,
    compute_f1_precision_recall,
    compute_kl_divergence,
    compute_knowledge_retention_score,
    EnhancedAssertionEvaluator,
    fast_evaluate
)

# Additional imports for fast evaluation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import AssertionDataset

class MockTokenizer:
    """Mock tokenizer for compatibility with EnhancedAssertionEvaluator."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

def load_teacher_data(file_path: str) -> List[Dict]:
    """Load teacher data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    # Skip header lines
                    if 'header' in item:
                        continue
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
    return data

def evaluate_model(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate a model's predictions using comprehensive metrics.

    Args:
        predictions: List of predicted assertions
        references: List of reference assertion lists

    Returns:
        Dictionary of metric name to value
    """
    if len(predictions) != len(references):
        print(f"Warning: Prediction count ({len(predictions)}) != reference count ({len(references)})")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

    # Initialize results
    results = {}

    # Create mock tokenizer for evaluator
    mock_tokenizer = MockTokenizer()
    evaluator = EnhancedAssertionEvaluator(mock_tokenizer)

    # Basic metrics
    f1, precision, recall = compute_f1_precision_recall(predictions, references)
    results['f1'] = f1
    results['precision'] = precision
    results['recall'] = recall

    # AST validity rate
    results['ast_validity'] = evaluator.evaluate_ast_validity(predictions)

    # PANS score (semantic equivalence)
    results['pans'] = compute_pans_score(predictions, references)

    # CodeBLEU score
    codebleu_scores = []
    for pred, refs in zip(predictions, references):
        score = compute_codebleu(refs, pred)
        codebleu_scores.append(score)
    results['codebleu'] = sum(codebleu_scores) / len(codebleu_scores) if codebleu_scores else 0.0

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate assertion generation models")
    parser.add_argument('--teacher_data', required=True,
                       help='Path to teacher data JSONL file (should contain ground truth)')
    parser.add_argument('--student_model_path',
                       help='Path to the trained student model directory')
    parser.add_argument('--student_limit', type=int, default=None,
                       help='Maximum number of validation examples to use for student evaluation')
    parser.add_argument('--device', default='cpu',
                       help='Device for computation (cpu/cuda)')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for KL divergence computation')

    args = parser.parse_args()

    print("🔍 Loading validation data...")
    validation_data = load_teacher_data(args.teacher_data)

    if not validation_data:
        print("❌ No valid validation data found")
        return

    print(f"📊 Loaded {len(validation_data)} validation examples")

    # Extract teacher predictions and ground truth references from the validation data
    teacher_predictions = []
    ground_truth_references = []

    for item in validation_data:
        # Get teacher's predicted assertions from the file
        teacher_pred = item.get('predicted_assertions', [])
        if isinstance(teacher_pred, list):
            teacher_pred = '\n'.join(teacher_pred)
        teacher_predictions.append(str(teacher_pred))

        # Get ground truth references from the 'assertions' key
        refs = item.get('assertions', [])
        if not isinstance(refs, list):
            refs = [str(refs)]
        ground_truth_references.append(refs)

    print("\n=== TEACHER MODEL EVALUATION ===")
    teacher_results = evaluate_model(teacher_predictions, ground_truth_references)
    print(f"AST Validity Rate: {teacher_results['ast_validity']*100:.1f}%")
    print(f"Semantic Equivalence Score (PANS): {teacher_results['pans']:.3f}")
    print(f"CodeBLEU Score: {teacher_results['codebleu']:.3f}")
    print(f"F1-Score: {teacher_results['f1']:.3f}")
    print(f"Precision: {teacher_results['precision']:.3f}")
    print(f"Recall: {teacher_results['recall']:.3f}")

    # Student evaluation if a model path is provided
    if args.student_model_path:
        # Load student model and tokenizer
        print(f"\n🧠 Loading student model from: {args.student_model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
        model.eval()
        if args.device == 'cuda' and torch.cuda.is_available():
            print("   -> Moving model to CUDA device.")
            model.to('cuda')
        else:
            print("   -> Using CPU device.")
            args.device = 'cpu'

        # Build validation dataset matching the distillation pipeline
        val_ds = AssertionDataset(args.teacher_data, tokenizer, max_samples=args.student_limit)
        # Insert student predictions loading print
        print("\n🔍 Loading student predictions...")
        print(f"📊 Loaded {len(val_ds)} student predictions")
        
        # Run the pipeline's fast evaluation to reproduce metrics_summary
        output_dir = os.path.join(os.path.dirname(__file__), 'post_hoc_evaluation')
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = fast_evaluate(
            model,
            tokenizer,
            val_ds,
            out_dir=output_dir,
            device=args.device,
            use_enhanced_metrics=True
        )
        print("\n=== STUDENT MODEL EVALUATION ===")
        print(f"AST Validity Rate: {metrics['ast_validity']*100:.1f}%")
        print(f"Semantic Equivalence Score (PANS): {metrics['pans']:.3f}")
        print(f"CodeBLEU Score: {metrics['codebleu']:.7f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        
        # === DETAILED COMPARISON SUMMARY ===
        print("\n=== DETAILED COMPARISON SUMMARY ===")
        # Performance Metrics
        print("📈 Performance Metrics:")
        for key, label in [('f1', 'F1-Score'), ('precision', 'Precision'), ('recall', 'Recall')]:
            t = teacher_results[key]
            s = metrics[key]
            gap = (s - t) * 100
            print(f"   ├─ {label} Gap: {gap:+.1f}% ({s:.3f} vs {t:.3f})")
        
        # Code Quality Metrics
        print("\n🔧 Code Quality Metrics:")
        # AST validity as percentage
        t_ast, s_ast = teacher_results['ast_validity'], metrics['ast_validity']
        gap_ast = (s_ast - t_ast) * 100
        print(f"   ├─ AST Validity Gap: {gap_ast:+.1f}% ({s_ast*100:.1f}% vs {t_ast*100:.1f}%)")
        for key in ['codebleu', 'pans']:
            t, s = teacher_results[key], metrics[key]
            gap = s - t
            print(f"   ├─ {key.upper()} Gap: {gap:+.3f} ({s:.3f} vs {t:.3f})")
        
        # Knowledge Transfer Assessment
        print("\n🎯 Knowledge Transfer Assessment:")
        krs = metrics.get('krs', metrics.get('knowledge_retention_score', 0.0))
        print(f"   ├─ Overall KRS: {krs*100:.1f}%")
        efficiency = krs * 100
        emoji = "💀 Very Poor (<30%)" if efficiency < 30 else "👍 Good (>=30%)"
        print(f"   ├─ Transfer Efficiency: {emoji}")
        print("   └─ Recommendations:")
        print("      • Consider increasing distillation temperature or alpha")
        print("      • Improve semantic similarity training")
        print("      • Review overall distillation strategy")
        
        # Export metrics JSON
        results_dict = {'teacher': teacher_results, 'student': metrics,
                        'gap': {k: (metrics[k] - teacher_results[k]) for k in teacher_results}}
        
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"\n💾 Detailed results exported to: {metrics_file}")
        return

if __name__ == '__main__':
    main()
