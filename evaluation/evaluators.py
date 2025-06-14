"""
Evaluation utilities for assertion generation.
"""
import os
import json
import csv
import math
import re
import difflib
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import javalang
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

# Enhanced metrics imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from codebleu import calc_codebleu
except ImportError:
    calc_codebleu = None
    print("⚠️ Warning: Official codebleu package not installed. Install with `pip install codebleu` to enable CodeBLEU scoring.")


def optimized_collate_fn(batch):
    """Collate function with better error handling."""
    if not batch:
        return {
            'input_ids': torch.empty(0, 0, dtype=torch.long),
            'attention_mask': torch.empty(0, 0, dtype=torch.long),
            'labels': torch.empty(0, 0, dtype=torch.long),
            'teacher_logits': torch.empty(0, 0, 0, dtype=torch.float)
        }
        
    ids = [x['input_ids'] for x in batch]
    masks = [x['attention_mask'] for x in batch]
    labs = [x['labels'] for x in batch]
    teacher_logits_list = [x['teacher_logits'] for x in batch]

    # Ensure tensors are at least 1D
    ids = [t.unsqueeze(0) if t.ndim == 0 else t for t in ids]
    masks = [t.unsqueeze(0) if t.ndim == 0 else t for t in masks]
    labs = [torch.tensor([-100]) if t.ndim == 0 else t for t in labs]

    # Pad sequences
    input_ids_padded = pad_sequence(ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labs, batch_first=True, padding_value=-100)

    # Process teacher logits
    max_seq_len = labels_padded.size(1)
    first_logits = teacher_logits_list[0]
    
    if first_logits.ndim == 3 and first_logits.size(0) == 1:
        vocab_size = first_logits.size(2)
    elif first_logits.ndim == 2:
        vocab_size = first_logits.size(1)
    else:
        raise ValueError(f"Unexpected teacher_logits shape: {first_logits.shape}")

    teacher_batch = []
    for t_logits in teacher_logits_list:
        if t_logits.ndim == 3 and t_logits.size(0) == 1:
            t_logits = t_logits.squeeze(0)
        
        current_seq_len = t_logits.size(0)
        if current_seq_len >= max_seq_len:
            processed = t_logits[:max_seq_len, :]
        else:
            pad_size = max_seq_len - current_seq_len
            padding = torch.zeros(pad_size, vocab_size, dtype=t_logits.dtype)
            processed = torch.cat([t_logits, padding], dim=0)
        
        teacher_batch.append(processed)

    teacher_logits_padded = torch.stack(teacher_batch, dim=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'teacher_logits': teacher_logits_padded
    }


# Primary CodeBLEU computation (recommended for code)
def compute_codebleu(references, hypothesis: str) -> float:
    """
    Compute CodeBLEU score using the official codebleu package (k4black/codebleu).
    
    Due to tree-sitter compatibility issues, this uses the BLEU component of CodeBLEU
    which provides code-aware tokenization without requiring AST parsing.
    
    Args:
        references: List of reference strings or list of reference lists
        hypothesis: Generated hypothesis string
        
    Returns:
        float: CodeBLEU-based BLEU score (0.0-1.0), where higher is better
              Returns 0.0 if CodeBLEU fails (no NLTK BLEU fallback)
    """
    if calc_codebleu is None:
        return 0.0
    
    # Clean and validate inputs
    if not hypothesis or not hypothesis.strip():
        return 0.0
        
    # Ensure proper reference format
    if isinstance(references, str):
        references = [references]
    
    # Handle nested list structure
    if isinstance(references, list) and len(references) > 0:
        if isinstance(references[0], list):
            # Already in correct format: [[ref1, ref2], [ref3, ref4], ...]
            clean_references = [[ref.strip() for ref in ref_list if ref and ref.strip()] 
                              for ref_list in references]
        else:
            # Convert to nested format: [ref1, ref2] -> [[ref1, ref2]]
            clean_references = [[ref.strip() for ref in references if ref and ref.strip()]]
    else:
        return 0.0
    
    # Filter out empty reference lists
    clean_references = [ref_list for ref_list in clean_references if ref_list]
    if not clean_references:
        return 0.0
    
    clean_hypothesis = hypothesis.strip()
    if not clean_hypothesis:
        return 0.0
    
    # Use CodeBLEU's BLEU component directly (avoids tree-sitter issues)
    try:
        from codebleu.bleu import corpus_bleu
        
        # Prepare references in correct format for corpus_bleu
        list_of_references = clean_references[0]  # Take first reference set
        hypotheses = [clean_hypothesis]
        
        # Validate inputs before calling corpus_bleu
        if not list_of_references or not hypotheses[0]:
            return 0.0
        
        # Calculate BLEU component using CodeBLEU's code-aware tokenization
        bleu_score = corpus_bleu([list_of_references], hypotheses)
        
        # Ensure we return a valid float
        if bleu_score is None or not isinstance(bleu_score, (int, float)):
            return 0.0
            
        return max(0.0, min(1.0, float(bleu_score)))  # Clamp to [0, 1]
        
    except Exception:
        # If even the manual approach fails, return 0.0 (no NLTK fallback)
        return 0.0

# Secondary BLEU computation (traditional)
def compute_bleu(references, hypothesis):
    """
    Compute traditional BLEU score for text similarity evaluation.
    
    Args:
        references: List of reference strings or single string
        hypothesis: Generated hypothesis string
    
    Returns:
        BLEU score (0.0-1.0)
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    if isinstance(references, str):
        references = [references]
    refs_tokens = [r.split() for r in references if r]
    hyp_tokens = hypothesis.split()
    try:
        return sentence_bleu(refs_tokens, hyp_tokens, smoothing_function=SmoothingFunction().method1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def compute_pans_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Compute Position-Aware N-gram Similarity (PANS) score for semantic equivalence.
    
    This function calculates n-gram similarity with position-aware weights,
    normalizing common Java assertion patterns before comparison.
    
    Args:
        predictions: List of predicted assertion strings
        references: List of reference assertion lists
        
    Returns:
        PANS score (0.0-1.0, higher is better)
    """
    if len(predictions) != len(references):
        return 0.0
    
    def normalize_assertion(assertion: str) -> str:
        """Normalize Java assertion patterns for better comparison."""
        # Remove common prefixes and normalize patterns
        normalized = assertion.strip()
        
        # Normalize assertion method patterns
        patterns = [
            (r'Assert\.assertEquals\s*\(', 'assertEquals('),
            (r'Assert\.assertTrue\s*\(', 'assertTrue('),
            (r'Assert\.assertFalse\s*\(', 'assertFalse('),
            (r'Assert\.assertNull\s*\(', 'assertNull('),
            (r'Assert\.assertNotNull\s*\(', 'assertNotNull('),
            (r'assertThat\s*\(\s*([^,]+)\s*\)\s*\.isEqualTo\s*\(', r'assertEquals(\1, '),
            (r'assertThat\s*\(\s*([^,]+)\s*,\s*is\s*\(([^)]+)\)\s*\)', r'assertEquals(\1, \2)'),
        ]
        
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def compute_ngram_similarity(pred: str, ref: str, max_n: int = 4) -> float:
        """Compute position-aware n-gram similarity."""
        pred_norm = normalize_assertion(pred)
        ref_norm = normalize_assertion(ref)
        
        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for n in range(1, min(max_n + 1, max(len(pred_tokens), len(ref_tokens)) + 1)):
            # Generate n-grams
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
            
            if not pred_ngrams or not ref_ngrams:
                continue
            
            # Calculate position-aware weights W_j = exp(-0.1 * j)
            matches = 0
            for i, pred_ngram in enumerate(pred_ngrams):
                for j, ref_ngram in enumerate(ref_ngrams):
                    if pred_ngram == ref_ngram:
                        # Position-aware weight
                        weight = math.exp(-0.1 * abs(i - j))
                        matches += weight
                        break
            
            # Normalize by maximum possible matches
            max_matches = min(len(pred_ngrams), len(ref_ngrams))
            if max_matches > 0:
                ngram_score = matches / max_matches
                weight = n  # Higher n-grams get more weight
                total_score += ngram_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    total_pans = 0.0
    for pred, refs in zip(predictions, references):
        if not refs:
            continue
        
        # Take the best score across all references
        best_score = max(compute_ngram_similarity(pred, ref) for ref in refs)
        total_pans += best_score
    
    return total_pans / len(predictions) if predictions else 0.0


def compute_f1_precision_recall(predictions: List[str], references: List[List[str]]) -> Tuple[float, float, float]:
    """
    Compute F1, precision, and recall scores using exact line/assertion matching.
    
    Args:
        predictions: List of predicted assertion strings
        references: List of reference assertion lists
        
    Returns:
        Tuple of (f1, precision, recall) with 6 decimal places precision
    """
    if len(predictions) != len(references):
        return 0.0, 0.0, 0.0
    
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    
    for pred, refs in zip(predictions, references):
        # Split predictions and references into individual assertions/lines
        pred_lines = [line.strip() for line in pred.split('\n') if line.strip()]
        
        # Collect all reference lines into a single set
        ref_lines_set = set()
        for ref in refs:
            ref_lines = [line.strip() for line in ref.split('\n') if line.strip()]
            ref_lines_set.update(ref_lines)
        
        pred_lines_set = set(pred_lines)
        
        # Calculate TP, FP, FN for this example using exact line matching
        tp = len(pred_lines_set & ref_lines_set)  # Intersection: correct predictions
        fp = len(pred_lines_set - ref_lines_set)  # Predicted but not in reference
        fn = len(ref_lines_set - pred_lines_set)  # In reference but not predicted
        
        total_tp += float(tp)
        total_fp += float(fp)
        total_fn += float(fn)
    
    # Calculate metrics with high precision
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Return with 6 decimal places precision
    return round(f1, 6), round(precision, 6), round(recall, 6)


def compute_kl_divergence(teacher_logits: np.ndarray, student_logits: np.ndarray,
                         temperature: float = 2.0) -> float:
    """
    Compute KL divergence between teacher and student logits with improved numerical stability.

    Args:
        teacher_logits: Teacher model logits
        student_logits: Student model logits
        temperature: Temperature for softmax computation

    Returns:
        KL divergence score (lower is better)
    """
    try:
        # Ensure logits have compatible shapes
        if teacher_logits.shape != student_logits.shape:
            # Try to align shapes by taking minimum dimensions
            min_shape = [min(t, s) for t, s in zip(teacher_logits.shape, student_logits.shape)]
            teacher_logits = teacher_logits[:min_shape[0], :min_shape[1], :min_shape[2]] if len(min_shape) == 3 else teacher_logits
            student_logits = student_logits[:min_shape[0], :min_shape[1], :min_shape[2]] if len(min_shape) == 3 else student_logits

        # Apply temperature scaling
        teacher_scaled = teacher_logits / temperature
        student_scaled = student_logits / temperature

        # Use more numerically stable softmax computation
        def stable_softmax(x):
            # Subtract max for numerical stability
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        teacher_probs = stable_softmax(teacher_scaled)
        student_probs = stable_softmax(student_scaled)

        # Compute KL divergence: KL(teacher || student)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12

        # Clip probabilities to avoid numerical issues
        teacher_probs = np.clip(teacher_probs, epsilon, 1.0 - epsilon)
        student_probs = np.clip(student_probs, epsilon, 1.0 - epsilon)

        kl_div = np.sum(teacher_probs * np.log(teacher_probs / student_probs), axis=-1)

        # Return mean KL divergence across all sequences
        mean_kl = float(np.mean(kl_div))

        # Clamp to reasonable range
        return min(mean_kl, 50.0)  # Cap at 50 to avoid extreme values

    except Exception as e:
        print(f"Error computing KL divergence: {e}")
        return float('inf')


def compute_knowledge_retention_score(teacher_logits: Optional[np.ndarray],
                                    student_logits: Optional[np.ndarray],
                                    teacher_f1: float, student_f1: float,
                                    temperature: float = 2.0) -> float:
    """
    Compute Knowledge Retention Score (KRS) combining output agreement and task performance.

    Formula: KRS = 0.6 * (1 - KL_Divergence_Normalized) + 0.4 * (Student_F1 / Teacher_F1)

    Args:
        teacher_logits: Teacher model logits
        student_logits: Student model logits
        teacher_f1: Teacher F1 score
        student_f1: Student F1 score
        temperature: Temperature for KL divergence computation

    Returns:
        KRS score (0.0-1.0, higher is better)
    """
    # Task Performance Preservation component
    if teacher_f1 > 0:
        performance_ratio = min(student_f1 / teacher_f1, 1.0)  # Cap at 1.0
    else:
        performance_ratio = 0.0

    # Output Agreement component
    if teacher_logits is not None and student_logits is not None:
        kl_div = compute_kl_divergence(teacher_logits, student_logits, temperature)
        # Normalize KL divergence (assume max reasonable KL is 10.0)
        kl_normalized = min(kl_div / 10.0, 1.0)
        output_agreement = 1.0 - kl_normalized
    else:
        print("Warning: Could not compute KL divergence, using performance ratio only")
        output_agreement = performance_ratio  # Fallback to performance ratio

    # Combine components
    krs = 0.6 * output_agreement + 0.4 * performance_ratio
    return round(krs, 6)


class EnhancedAssertionEvaluator:
    """
    Comprehensive evaluation system for assertion generation with code-specific metrics.
    
    Provides detailed analysis of generated assertions including:
    - CodeBLEU (primary code metric)
    - PANS (Position-Aware N-gram Similarity)
    - AST validity (syntactic correctness)
    - Semantic similarity (custom code-aware)
    - Token accuracy (exact matching)
    - F1, Precision, Recall
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metrics = ['codebleu', 'bleu', 'ast_validity', 'semantic_similarity', 'token_accuracy', 'pans', 'f1', 'precision', 'recall']
    
    def evaluate_ast_validity(self, predictions: List[str]) -> float:
        """
        Evaluate the percentage of predictions that are syntactically valid Java.
        
        Args:
            predictions: List of generated assertion strings
            
        Returns:
            Fraction of syntactically valid predictions
        """
        if not AST_AVAILABLE:
            return 0.0
            
        valid_count = 0
        total_count = len(predictions)
        
        for pred in predictions:
            try:
                # Wrap prediction in minimal Java class structure
                java_code = f"""
                public class TestClass {{
                    public void testMethod() {{
                        {pred}
                    }}
                }}
                """
                
                # Try to parse
                javalang.parse.parse(java_code)
                valid_count += 1
                
            except Exception:
                continue
                
        return valid_count / total_count if total_count > 0 else 0.0
    
    def evaluate_semantic_similarity(self, predictions: List[str], 
                                   references: List[List[str]]) -> float:
        """
        Evaluate semantic similarity using token overlap and structural analysis.
        
        This is a code-specific semantic similarity that considers:
        - Variable name similarity
        - Method call patterns
        - Assertion structure
        """
        if len(predictions) != len(references):
            return 0.0
            
        total_similarity = 0.0
        
        for pred, refs in zip(predictions, references):
            max_sim = 0.0
            
            for ref in refs:
                # Token-level similarity
                pred_tokens = set(pred.split())
                ref_tokens = set(ref.split())
                
                if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                    sim = 1.0
                elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    sim = 0.0
                else:
                    intersection = len(pred_tokens & ref_tokens)
                    union = len(pred_tokens | ref_tokens)
                    sim = intersection / union if union > 0 else 0.0
                
                # Boost similarity for common assertion patterns
                assertion_patterns = ['assert', 'assertEquals', 'assertTrue', 'assertFalse', 'assertNull']
                pred_patterns = sum(1 for pattern in assertion_patterns if pattern in pred)
                ref_patterns = sum(1 for pattern in assertion_patterns if pattern in ref)
                
                if pred_patterns > 0 and ref_patterns > 0:
                    pattern_sim = min(pred_patterns, ref_patterns) / max(pred_patterns, ref_patterns)
                    sim = 0.7 * sim + 0.3 * pattern_sim
                
                max_sim = max(max_sim, sim)
            
            total_similarity += max_sim
        
        return total_similarity / len(predictions) if len(predictions) > 0 else 0.0
    
    def evaluate_token_accuracy(self, predictions: List[str], 
                               references: List[List[str]]) -> float:
        """
        Token-level accuracy considering exact matches and partial matches with high precision.
        """
        if len(predictions) != len(references):
            return 0.0
            
        total_accuracy = 0.0
        valid_comparisons = 0
        
        for pred, refs in zip(predictions, references):
            if not refs:
                continue
                
            max_acc = 0.0
            
            for ref in refs:
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(ref_tokens) == 0 and len(pred_tokens) == 0:
                    acc = 1.0
                elif len(ref_tokens) == 0:
                    acc = 0.0
                else:
                    # Calculate token-level accuracy with position matching
                    matches = 0
                    min_len = min(len(pred_tokens), len(ref_tokens))
                    
                    for i in range(min_len):
                        if pred_tokens[i] == ref_tokens[i]:
                            matches += 1
                    
                    # Penalty for length difference
                    length_penalty = min(len(pred_tokens), len(ref_tokens)) / max(len(pred_tokens), len(ref_tokens))
                    acc = (matches / len(ref_tokens)) * length_penalty
                
                max_acc = max(max_acc, acc)
            
            total_accuracy += max_acc
            valid_comparisons += 1
        
        return total_accuracy / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def evaluate_batch(self, predictions: List[str], references: List[List[str]], 
                      contexts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of a batch of predictions with enhanced code metrics.
        
        Args:
            predictions: List of generated assertions
            references: List of reference assertion lists
            contexts: Optional list of code contexts
            
        Returns:
            Dictionary of metric name to value (all with high precision)
        """
        metrics = {}
        
        if len(predictions) == 0:
            return {metric: 0.0 for metric in self.metrics + ['code_quality_score']}
        
        # Primary Code Metrics (most important)
        codebleu_scores = []
        
        # Secondary Metrics
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            # CodeBLEU (primary code metric)
            codebleu_scores.append(compute_codebleu(refs, pred))
            
            # Standard BLEU (secondary)
            bleu_scores.append(compute_bleu(refs, pred))
        
        # Average scores with high precision
        metrics['codebleu'] = round(sum(codebleu_scores) / len(codebleu_scores), 6)
        metrics['bleu'] = round(sum(bleu_scores) / len(bleu_scores), 6)
        
        # Core evaluation metrics
        metrics['ast_validity'] = round(self.evaluate_ast_validity(predictions), 6)
        metrics['semantic_similarity'] = round(self.evaluate_semantic_similarity(predictions, references), 6)
        metrics['token_accuracy'] = round(self.evaluate_token_accuracy(predictions, references), 6)
        
        # Enhanced metrics from evaluate_assertions.py
        metrics['pans'] = round(compute_pans_score(predictions, references), 6)
        f1, precision, recall = compute_f1_precision_recall(predictions, references)
        metrics['f1'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Additional debugging metrics
        total_pred_tokens = sum(len(pred.split()) for pred in predictions)
        total_ref_tokens = sum(len(refs[0].split()) if refs else 0 for refs in references)
        exact_matches = sum(1 for pred, refs in zip(predictions, references) 
                           if pred.strip() in [ref.strip() for ref in refs])
        
        metrics['exact_match_ratio'] = round(exact_matches / len(predictions), 6)
        metrics['avg_prediction_length'] = round(total_pred_tokens / len(predictions), 2)
        metrics['avg_reference_length'] = round(total_ref_tokens / len(references), 2)
        
        # Enhanced code quality score (weighted combination emphasizing code-specific metrics)
        quality_score = (
            0.30 * metrics['codebleu'] +           # Primary: CodeBLEU for code generation
            0.20 * metrics['ast_validity'] +       # Critical: syntactic correctness
            0.20 * metrics['pans'] +               # Important: semantic equivalence
            0.15 * metrics['f1'] +                 # Important: overall performance
            0.10 * metrics['semantic_similarity'] + # Supporting: semantic matching
            0.05 * metrics['token_accuracy']       # Supporting: exact matching
        )
        metrics['code_quality_score'] = round(quality_score, 6)
        
        return metrics


@torch.no_grad()
def fast_evaluate(model, tokenizer, dataset, out_dir, device=None, epoch=None, use_enhanced_metrics=True):
    """Enhanced evaluation function with comprehensive metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    results, predictions = [], []
    
    # Initialize enhanced evaluator
    enhanced_evaluator = EnhancedAssertionEvaluator(tokenizer) if use_enhanced_metrics else None
    
    eval_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=optimized_collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    max_gen_len = getattr(dataset, 'max_out', 128)
    
    # Collect all predictions and references for batch evaluation
    all_predictions = []
    all_references = []
    
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f'Eval {epoch or "Final"}')):
        inp = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        
        generated = model.generate(
            inp,
            attention_mask=mask,
            max_length=max_gen_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1
        )

        for i in range(generated.size(0)):
            item_idx = batch_idx * eval_loader.batch_size + i
            if item_idx >= len(dataset):
                continue
                
            original_data = dataset.samples_data[item_idx]
            target_key = 'original_target' if 'original_target' in original_data else 'assertions'
            refs = original_data.get(target_key, [])
            
            if not isinstance(refs, list):
                refs = [str(refs)]

            hyp = tokenizer.decode(generated[i], skip_special_tokens=True)
            pred_lines = [l.strip() for l in hyp.split('\n') if l.strip()]
            ref_set = set(str(r) for r in refs)
            
            # Traditional metrics
            tp = len(set(pred_lines) & ref_set)
            precision = tp / len(pred_lines) if pred_lines else 0.0
            recall = tp / len(ref_set) if ref_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            similarity = difflib.SequenceMatcher(None, ' '.join(pred_lines), ' '.join(refs)).ratio()
            codebleu = compute_codebleu(refs, hyp)

            results.append({
                'id': item_idx, 'precision': precision, 'recall': recall,
                'f1': f1, 'accuracy': recall, 'similarity': similarity,
                'codebleu': codebleu, 'exact_match_count': tp
            })
            
            predictions.append({
                'id': item_idx, 'prediction': hyp, 'references': refs,
                'epoch': epoch or 'final'
            })
            
            # Collect for enhanced evaluation
            if use_enhanced_metrics and enhanced_evaluator:
                all_predictions.append(hyp)
                all_references.append(refs)

    # Compute enhanced metrics
    enhanced_metrics = {}
    if use_enhanced_metrics and enhanced_evaluator and len(all_predictions) > 0:
        try:
            enhanced_metrics = enhanced_evaluator.evaluate_batch(all_predictions, all_references)
        except Exception as e:
            print(f"Warning: Enhanced metrics computation failed: {e}")

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_epoch{epoch}" if epoch else "_final"
    
    with open(os.path.join(out_dir, f'predictions{suffix}.jsonl'), 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    if results:
        with open(os.path.join(out_dir, f'metrics{suffix}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        summary = {k: round(sum(r[k] for r in results) / len(results), 3) 
                  for k in ['precision', 'recall', 'f1', 'accuracy', 'similarity', 'codebleu']}
        summary['exact_match_total'] = sum(r['exact_match_count'] for r in results)
        summary['epoch'] = epoch or 'final'
        
        # Add enhanced metrics to summary with 3 decimal precision
        for metric, value in enhanced_metrics.items():
            summary[metric] = round(value, 3) if isinstance(value, float) else value

        # Save sample predictions for debugging
        sample_predictions_file = os.path.join(out_dir, f'sample_predictions{suffix}.txt')
        with open(sample_predictions_file, 'w') as f:
            f.write("=== SAMPLE PREDICTIONS FOR DEBUGGING ===\n\n")
            num_samples = min(10, len(all_predictions))
            for i in range(num_samples):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Prediction: {all_predictions[i]}\n")
                f.write(f"Reference: {all_references[i][0] if all_references[i] else 'N/A'}\n")
                f.write(f"Pred tokens: {len(all_predictions[i].split())}\n")
                f.write(f"Ref tokens: {len(all_references[i][0].split()) if all_references[i] else 0}\n")
                f.write("-" * 80 + "\n")
            
            # Add overall statistics
            f.write("\n=== OVERALL STATISTICS ===\n")
            f.write(f"Total predictions: {len(all_predictions)}\n")
            
            # Safe division with checks for empty lists
            if all_predictions:
                avg_pred_len = sum(len(p.split()) for p in all_predictions) / len(all_predictions)
                f.write(f"Average prediction length: {avg_pred_len:.2f} tokens\n")
            else:
                f.write("Average prediction length: N/A (no predictions)\n")
                
            if all_references:
                avg_ref_len = sum(len(r[0].split()) if r else 0 for r in all_references) / len(all_references)
                f.write(f"Average reference length: {avg_ref_len:.2f} tokens\n")
            else:
                f.write("Average reference length: N/A (no references)\n")
                
            f.write(f"Empty predictions: {sum(1 for p in all_predictions if not p.strip())}\n")
            f.write(f"Empty references: {sum(1 for r in all_references if not r or not r[0].strip())}\n")

        with open(os.path.join(out_dir, f'metrics_summary{suffix}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    return {}
