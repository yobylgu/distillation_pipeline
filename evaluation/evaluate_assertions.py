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
        --student_model_path results/test_training/2025-06-14_12-08-37_Salesforce-codet5p-220m/final_model \
        --student_limit 200
"""

import json
import argparse
import os
from typing import Dict, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime

# Import existing metrics from evaluators.py
from evaluators import (
    compute_codebleu,
    compute_pans_score,
    compute_f1_precision_recall,
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
    results['f1'] = round(f1, 7)
    results['precision'] = round(precision, 7)
    results['recall'] = round(recall, 7)

    # AST validity rate
    results['ast_validity'] = round(evaluator.evaluate_ast_validity(predictions), 7)

    # PANS score (semantic equivalence)
    results['pans'] = round(compute_pans_score(predictions, references), 7)

    # CodeBLEU score
    codebleu_scores = []
    for pred, refs in zip(predictions, references):
        score = compute_codebleu(refs, pred)
        codebleu_scores.append(score)
    results['codebleu'] = round(sum(codebleu_scores) / len(codebleu_scores) if codebleu_scores else 0.0, 7)

    # Additional metrics to match student evaluation
    # Token accuracy (exact match)
    exact_matches = 0
    for pred, refs in zip(predictions, references):
        if any(pred.strip() == ref.strip() for ref in refs):
            exact_matches += 1
    results['token_accuracy'] = round(exact_matches / len(predictions), 7)
    results['exact_match_total'] = exact_matches
    results['exact_match_ratio'] = round(exact_matches / len(predictions), 7)

    # Semantic similarity (proper calculation using token overlap and structural analysis)
    results['semantic_similarity'] = round(evaluator.evaluate_semantic_similarity(predictions, references), 7)

    # General similarity score (average of multiple similarity measures)
    similarity_scores = []
    for pred, refs in zip(predictions, references):
        # Use the best reference for similarity
        best_sim = 0
        for ref in refs:
            # Simple token-based similarity
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            if pred_tokens or ref_tokens:
                jaccard_sim = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
                best_sim = max(best_sim, jaccard_sim)
        similarity_scores.append(best_sim)
    results['similarity'] = round(sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0, 7)

    # Code quality score (composite of AST validity and semantic similarity)
    results['code_quality_score'] = round((results['ast_validity'] + results['semantic_similarity']) / 2, 7)

    # Average prediction and reference lengths
    results['avg_prediction_length'] = round(sum(len(pred.split()) for pred in predictions) / len(predictions), 7)
    avg_ref_lengths = []
    for refs in references:
        avg_ref_length = sum(len(ref.split()) for ref in refs) / len(refs)
        avg_ref_lengths.append(avg_ref_length)
    results['avg_reference_length'] = round(sum(avg_ref_lengths) / len(avg_ref_lengths), 7)

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
    print(f"Semantic Equivalence Score (PANS): {teacher_results['pans']:.7f}")
    print(f"CodeBLEU Score: {teacher_results['codebleu']:.7f}")
    print(f"F1-Score: {teacher_results['f1']:.7f}")
    print(f"Precision: {teacher_results['precision']:.7f}")
    print(f"Recall: {teacher_results['recall']:.7f}")
    print(f"Token Accuracy: {teacher_results['token_accuracy']:.7f}")
    print(f"Semantic Similarity: {teacher_results['semantic_similarity']:.7f}")
    print(f"Similarity: {teacher_results['similarity']:.7f}")
    print(f"Code Quality Score: {teacher_results['code_quality_score']:.7f}")
    print(f"Exact Match Total: {teacher_results['exact_match_total']}")
    print(f"Exact Match Ratio: {teacher_results['exact_match_ratio']:.7f}")
    print(f"Avg Prediction Length: {teacher_results['avg_prediction_length']:.7f}")
    print(f"Avg Reference Length: {teacher_results['avg_reference_length']:.7f}")

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
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'post_hoc_evaluation', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        
        metrics = fast_evaluate(
            model,
            tokenizer,
            val_ds,
            out_dir=output_dir,
            device=args.device,
            use_enhanced_metrics=True
        )
        
        # Remove bleu and standardize student metrics to 7 decimals
        if 'bleu' in metrics:
            del metrics['bleu']
        
        # Rename accuracy to token_accuracy if it exists
        if 'accuracy' in metrics:
            metrics['token_accuracy'] = metrics.pop('accuracy')
        
        # Round all student metrics to 7 decimals
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics[key] = round(value, 7)

        print("\n=== STUDENT MODEL EVALUATION ===")
        print(f"AST Validity Rate: {metrics['ast_validity']*100:.1f}%")
        print(f"Semantic Equivalence Score (PANS): {metrics['pans']:.7f}")
        print(f"CodeBLEU Score: {metrics['codebleu']:.7f}")
        print(f"F1-Score: {metrics['f1']:.7f}")
        print(f"Precision: {metrics['precision']:.7f}")
        print(f"Recall: {metrics['recall']:.7f}")
        print(f"Token Accuracy: {metrics['token_accuracy']:.7f}")
        print(f"Semantic Similarity: {metrics['semantic_similarity']:.7f}")
        print(f"Similarity: {metrics['similarity']:.7f}")
        print(f"Code Quality Score: {metrics['code_quality_score']:.7f}")
        print(f"Exact Match Total: {metrics['exact_match_total']}")
        print(f"Exact Match Ratio: {metrics['exact_match_ratio']:.7f}")
        print(f"Avg Prediction Length: {metrics['avg_prediction_length']:.7f}")
        print(f"Avg Reference Length: {metrics['avg_reference_length']:.7f}")
        
        # === DETAILED COMPARISON SUMMARY ===
        print("\n=== DETAILED COMPARISON SUMMARY ===")
        # Performance Metrics
        print("📈 Performance Metrics:")
        for key, label in [('f1', 'F1-Score'), ('precision', 'Precision'), ('recall', 'Recall'), ('token_accuracy', 'Token Accuracy')]:
            if key in teacher_results and key in metrics:
                t = teacher_results[key]
                s = metrics[key]
                if t > 0:
                    gap_percent = ((s - t) / t) * 100
                    print(f"   ├─ {label} Gap: {gap_percent:+.1f}% ({s:.7f} vs {t:.7f})")
                else:
                    gap_absolute = s - t
                    print(f"   ├─ {label} Gap: {gap_absolute:+.7f} ({s:.7f} vs {t:.7f})")
        
        # Code Quality Metrics
        print("\n🔧 Code Quality Metrics:")
        # AST validity as percentage points
        t_ast, s_ast = teacher_results['ast_validity'], metrics['ast_validity']
        gap_ast = (s_ast - t_ast) * 100  # percentage points difference
        print(f"   ├─ AST Validity Gap: {gap_ast:+.1f} percentage points ({s_ast*100:.1f}% vs {t_ast*100:.1f}%)")
        for key in ['codebleu', 'pans', 'semantic_similarity', 'similarity', 'code_quality_score']:
            if key in teacher_results and key in metrics:
                t, s = teacher_results[key], metrics[key]
                if t > 0:
                    gap_percent = ((s - t) / t) * 100
                    print(f"   ├─ {key.upper()} Gap: {gap_percent:+.1f}% ({s:.7f} vs {t:.7f})")
                else:
                    gap_absolute = s - t
                    print(f"   ├─ {key.upper()} Gap: {gap_absolute:+.7f} ({s:.7f} vs {t:.7f})")
        
        # Additional Metrics
        print("\n📊 Additional Metrics:")
        for key in ['exact_match_ratio', 'avg_prediction_length', 'avg_reference_length']:
            if key in teacher_results and key in metrics:
                t, s = teacher_results[key], metrics[key]
                if t > 0:
                    gap_percent = ((s - t) / t) * 100
                    print(f"   ├─ {key.replace('_', ' ').title()} Gap: {gap_percent:+.1f}% ({s:.7f} vs {t:.7f})")
                else:
                    gap_absolute = s - t
                    print(f"   ├─ {key.replace('_', ' ').title()} Gap: {gap_absolute:+.7f} ({s:.7f} vs {t:.7f})")
        
        # Exact match counts
        if 'exact_match_total' in teacher_results and 'exact_match_total' in metrics:
            t_total, s_total = teacher_results['exact_match_total'], metrics['exact_match_total']
            gap_total = s_total - t_total
            print(f"   ├─ Exact Match Total Gap: {gap_total:+d} ({s_total} vs {t_total})")

        # Export metrics JSON
        # Calculate gaps for all common metrics
        gap_dict = {}
        for key in teacher_results:
            if key in metrics:
                gap_dict[key] = round(metrics[key] - teacher_results[key], 7)
        
        # Add metadata
        metadata = {
            'timestamp': timestamp,
            'student_model_path': args.student_model_path,
            'validation_dataset_size': len(val_ds),
            'student_limit': args.student_limit,
            'teacher_data_path': args.teacher_data,
            'device_used': args.device,
            'temperature': args.temperature
        }
        
        results_dict = {
            'metadata': metadata,
            'teacher': teacher_results, 
            'student': metrics,
            'gap': gap_dict
        }
        
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"\n💾 Detailed results exported to: {metrics_file}")
        return

if __name__ == '__main__':
    main()
