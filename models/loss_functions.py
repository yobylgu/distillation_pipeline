"""
Loss functions for knowledge distillation.
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import javalang
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

# NEW: Import for semantic similarity loss
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def optimized_distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """Distillation loss with error handling."""
    if student_logits.size(-1) != teacher_logits.size(-1):
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

    active_mask = (labels != -100)
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    labels_flat = labels.view(-1)
    active_flat = active_mask.view(-1)

    active_student = student_flat[active_flat]
    active_teacher = teacher_flat[active_flat]
    active_labels = labels_flat[active_flat]

    ce_loss = F.cross_entropy(active_student, active_labels)
    teacher_probs = F.softmax(active_teacher / T, dim=-1)
    student_log_probs = F.log_softmax(active_student / T, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

    return alpha * ce_loss + (1 - alpha) * kl_loss

def optimized_distillation_loss_with_logging(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """Enhanced distillation loss with component tracking for logging."""
    if student_logits.size(-1) != teacher_logits.size(-1):
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

    active_mask = (labels != -100)
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), {
            'ce': 0.0, 'kl': 0.0, 'total': 0.0
        }

    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    labels_flat = labels.view(-1)
    active_flat = active_mask.view(-1)

    active_student = student_flat[active_flat]
    active_teacher = teacher_flat[active_flat]
    active_labels = labels_flat[active_flat]

    # Calculate individual loss components
    ce_loss = F.cross_entropy(active_student, active_labels)
    teacher_probs = F.softmax(active_teacher / T, dim=-1)
    student_log_probs = F.log_softmax(active_student / T, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    
    # Return loss components for logging
    loss_components = {
        'ce': ce_loss.item(),
        'kl': kl_loss.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_components

def compute_pans_loss(student_logits: torch.Tensor, labels: torch.Tensor, 
                      tokenizer, n_gram_sizes: List[int] = [1, 2, 3]) -> torch.Tensor:
    """
    Position-Aware N-gram Similarity (PANS) loss for code generation.
    
    Focuses on n-gram preservation which is crucial for code syntax and semantics.
    This loss function helps maintain syntactic correctness and semantic meaning
    even when exact token sequences vary.
    
    Args:
        student_logits: Student model predictions [batch, seq_len, vocab_size]
        labels: Ground truth labels [batch, seq_len]
        tokenizer: Tokenizer for converting ids to tokens
        n_gram_sizes: List of n-gram sizes to consider
        
    Returns:
        PANS loss value
    """
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Get predicted tokens
    predicted_ids = torch.argmax(student_logits, dim=-1)  # [batch, seq_len]
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for batch_idx in range(batch_size):
        pred_ids = predicted_ids[batch_idx]
        true_ids = labels[batch_idx]
        
        # Filter out padding tokens
        mask = (true_ids != -100)
        if mask.sum() == 0:
            continue
            
        pred_tokens = pred_ids[mask]
        true_tokens = true_ids[mask]
        
        # Convert to actual tokens for n-gram analysis
        try:
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)
            
            pred_words = pred_text.split()
            true_words = true_text.split()
            
            if len(pred_words) == 0 or len(true_words) == 0:
                continue
                
            # Calculate n-gram overlap for each n-gram size
            for n in n_gram_sizes:
                pred_ngrams = set()
                true_ngrams = set()
                
                # Generate n-grams
                for i in range(len(pred_words) - n + 1):
                    pred_ngrams.add(tuple(pred_words[i:i+n]))
                for i in range(len(true_words) - n + 1):
                    true_ngrams.add(tuple(true_words[i:i+n]))
                
                if len(true_ngrams) == 0:
                    continue
                    
                # Calculate overlap
                overlap = len(pred_ngrams & true_ngrams)
                precision = overlap / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
                recall = overlap / len(true_ngrams)
                
                # PANS penalty: 1 - F1 score of n-gram overlap
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    ngram_loss = 1.0 - f1
                else:
                    ngram_loss = 1.0
                
                total_loss = total_loss + torch.tensor(ngram_loss, device=device, requires_grad=True)
                
        except Exception as e:
            # Skip problematic sequences
            continue
    
    # Normalize by batch size and n-gram types
    if batch_size > 0 and len(n_gram_sizes) > 0:
        total_loss = total_loss / (batch_size * len(n_gram_sizes))
    
    return total_loss

def compute_ast_penalty(predictions: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    AST-aware penalty for Java code predictions.
    
    Penalizes predictions that would result in syntactically invalid Java code.
    This is particularly important for assertion generation.
    
    Args:
        predictions: Predicted token ids [batch, seq_len]
        tokenizer: Tokenizer for converting ids to text
        
    Returns:
        AST penalty value
    """
    device = predictions.device
    
    if not AST_AVAILABLE:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    total_penalty = 0.0
    batch_size = predictions.size(0)
    
    for batch_idx in range(batch_size):
        try:
            # Decode prediction to text
            pred_ids = predictions[batch_idx]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            # Try to parse as Java code
            # For assertions, we often need to wrap in a minimal class structure
            java_code = f"""
            public class TestClass {{
                public void testMethod() {{
                    {pred_text}
                }}
            }}
            """
            
            try:
                # Attempt to parse the Java code
                tree = javalang.parse.parse(java_code)
                # If parsing succeeds, no penalty
                ast_penalty = 0.0
            except javalang.parser.JavaSyntaxError:
                # If parsing fails, apply penalty
                ast_penalty = 1.0
            except Exception:
                # For other errors, apply moderate penalty
                ast_penalty = 0.5
                
        except Exception:
            # If decoding fails, apply penalty
            ast_penalty = 1.0
            
        total_penalty += ast_penalty
    
    # Normalize by batch size
    avg_penalty = total_penalty / batch_size if batch_size > 0 else 0.0
    return torch.tensor(avg_penalty, device=device, requires_grad=True)

def compute_weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                                  token_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional per-token weighting.
    
    NEW: Enhanced cross-entropy loss supporting token-specific weights for critical tokens (Task 4.3).
    
    Args:
        logits: Student model predictions [batch, seq_len, vocab_size]
        labels: Ground truth labels [batch, seq_len]
        token_weights: Per-token weights [vocab_size] for critical token weighting
        
    Returns:
        Cross-entropy loss value
    """
    device = logits.device
    
    # Handle padding tokens
    active_mask = (labels != -100)
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Flatten tensors
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    active_flat = active_mask.view(-1)
    
    # Get active (non-padded) tokens
    active_logits = logits_flat[active_flat]
    active_labels = labels_flat[active_flat]
    
    # Compute cross-entropy loss
    if token_weights is not None:
        # Ensure token_weights is on the same device
        if token_weights.device != device:
            token_weights = token_weights.to(device)
        
        # Use token weights as class weights in cross-entropy
        ce_loss = F.cross_entropy(active_logits, active_labels, weight=token_weights, reduction='mean')
    else:
        # Standard cross-entropy without weights
        ce_loss = F.cross_entropy(active_logits, active_labels)
    
    return ce_loss

def compute_focal_loss(logits: torch.Tensor, labels: torch.Tensor, 
                      gamma: float = 2.0, alpha: float = 0.25, 
                      token_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute Focal Loss for addressing class imbalance and hard examples.
    
    Focal Loss modulates the cross-entropy loss to focus learning on hard-to-classify examples
    while down-weighting easy examples. This is particularly useful for code generation where
    certain tokens (like common keywords) are easy to predict while complex assertion logic
    requires more attention.
    
    Args:
        logits: Student model predictions [batch, seq_len, vocab_size]
        labels: Ground truth labels [batch, seq_len]
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
        alpha: Weighting factor for rare class (typically 0.25)
        token_weights: NEW (Task 4.3) - Per-token weights [vocab_size] for critical token weighting
        
    Returns:
        Focal loss value
    """
    device = logits.device
    
    # Handle padding tokens
    active_mask = (labels != -100)
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Flatten tensors
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    active_flat = active_mask.view(-1)
    
    # Get active (non-padded) tokens
    active_logits = logits_flat[active_flat]
    active_labels = labels_flat[active_flat]
    
    # Compute cross-entropy
    ce_loss = F.cross_entropy(active_logits, active_labels, reduction='none')
    
    # Compute probabilities and focal weight
    log_pt = F.log_softmax(active_logits, dim=-1)
    pt = log_pt.gather(1, active_labels.unsqueeze(1)).squeeze(1)
    pt = pt.exp()
    
    # Apply focal loss formula: FL = -alpha * (1-pt)^gamma * log(pt)
    focal_weight = alpha * (1 - pt) ** gamma
    focal_loss = focal_weight * ce_loss
    
    # NEW: Apply token-specific weights (Task 4.3)
    if token_weights is not None:
        # Ensure token_weights is on the same device
        if token_weights.device != device:
            token_weights = token_weights.to(device)
        
        # Get token weights for the active labels
        label_weights = token_weights[active_labels]
        focal_loss = focal_loss * label_weights
    
    return focal_loss.mean()

def compute_jsd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                    temperature: float = 2.0) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence (JSD) loss for knowledge distillation.
    
    JSD is a symmetric and stable alternative to KL-Divergence that measures the
    similarity between two probability distributions. Unlike KL-divergence, JSD
    is bounded and symmetric, making training more stable.
    
    Args:
        student_logits: Student model predictions [batch, seq_len, vocab_size]
        teacher_logits: Teacher model predictions [batch, seq_len, vocab_size]
        temperature: Temperature for probability smoothing
        
    Returns:
        JSD loss value
    """
    device = student_logits.device
    
    # Ensure logits have the same vocabulary size
    if student_logits.size(-1) != teacher_logits.size(-1):
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]
    
    # Flatten to [batch * seq_len, vocab_size]
    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    
    # Compute softmax probabilities with temperature
    student_probs = F.softmax(student_flat / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_flat / temperature, dim=-1)
    
    # Compute midpoint distribution M = (P + Q) / 2
    midpoint = (student_probs + teacher_probs) / 2
    
    # Ensure numerical stability
    eps = 1e-8
    midpoint = torch.clamp(midpoint, min=eps)
    student_probs = torch.clamp(student_probs, min=eps)
    teacher_probs = torch.clamp(teacher_probs, min=eps)
    
    # Compute KL divergences from midpoint
    kl_student = F.kl_div(midpoint.log(), student_probs, reduction='batchmean')
    kl_teacher = F.kl_div(midpoint.log(), teacher_probs, reduction='batchmean')
    
    # JSD = 0.5 * (KL(P||M) + KL(Q||M))
    jsd_loss = 0.5 * (kl_student + kl_teacher)
    
    # Apply temperature scaling (similar to KL loss in traditional distillation)
    return jsd_loss * (temperature * temperature)

def compute_semantic_loss(student_logits: torch.Tensor, labels: torch.Tensor, 
                         tokenizer, sentence_transformer_model) -> torch.Tensor:
    """
    Compute semantic similarity loss between student predictions and ground truth.
    
    This loss function evaluates how semantically similar the generated assertions
    are to the ground truth, going beyond token-level accuracy to measure meaning
    preservation. It uses sentence transformers to encode both predictions and 
    references into semantic embeddings, then computes cosine similarity.
    
    Args:
        student_logits: Student model predictions [batch, seq_len, vocab_size]
        labels: Ground truth labels [batch, seq_len]
        tokenizer: Tokenizer for converting token IDs to text
        sentence_transformer_model: Pre-trained sentence transformer model
        
    Returns:
        Semantic similarity loss (1 - cosine_similarity)
    """
    device = student_logits.device
    
    # Check if sentence transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE or sentence_transformer_model is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    batch_size = student_logits.size(0)
    
    # Get predicted tokens
    predicted_ids = torch.argmax(student_logits, dim=-1)  # [batch, seq_len]
    
    total_similarity = 0.0
    valid_pairs = 0
    
    for batch_idx in range(batch_size):
        try:
            # Extract non-padded tokens
            pred_ids = predicted_ids[batch_idx]
            true_ids = labels[batch_idx]
            
            # Filter out padding tokens (-100)
            mask = (true_ids != -100)
            if mask.sum() == 0:
                continue
                
            pred_tokens = pred_ids[mask]
            true_tokens = true_ids[mask]
            
            # Decode to text
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)
            
            # Skip empty texts
            if not pred_text.strip() or not true_text.strip():
                continue
            
            # Encode with sentence transformer (disable progress bar)
            embeddings = sentence_transformer_model.encode([pred_text, true_text], convert_to_tensor=True, show_progress_bar=False)
            
            # Ensure embeddings are on the correct device
            embeddings = embeddings.to(device)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
            total_similarity += similarity.item()
            valid_pairs += 1
            
        except Exception as e:
            # Skip problematic sequences and continue
            continue
    
    if valid_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Average cosine similarity across batch
    avg_similarity = total_similarity / valid_pairs
    
    # Return loss as (1 - similarity) to minimize dissimilarity
    semantic_loss = 1.0 - avg_similarity
    return torch.tensor(semantic_loss, device=device, requires_grad=True)

def enhanced_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                             labels: torch.Tensor, tokenizer, T: float = 2.0, 
                             alpha: float = 0.5, pans_weight: float = 0.12) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enhanced distillation loss combining traditional CE+KL with PANS.
    
    This is a convenience function that provides the PANS enhancement
    while maintaining compatibility with existing code.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Ground truth labels
        tokenizer: Tokenizer for text processing
        T: Temperature for distillation
        alpha: Weight for CE vs KL loss
        pans_weight: Weight for PANS component
        
    Returns:
        total_loss: Combined loss value
        component_losses: Dictionary of loss components
    """
    # Compute base distillation loss
    base_loss, base_components = optimized_distillation_loss_with_logging(
        student_logits, teacher_logits, labels, T, alpha
    )
    
    # Add PANS component
    pans_loss = compute_pans_loss(student_logits, labels, tokenizer)
    
    # Combine losses
    total_loss = base_loss + pans_weight * pans_loss
    
    # Update component dictionary
    component_losses = base_components.copy()
    component_losses['pans'] = pans_loss.item() if hasattr(pans_loss, 'item') else float(pans_loss)
    component_losses['total'] = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
    
    return total_loss, component_losses

def compute_info_nce_loss(anchor_embeddings: torch.Tensor,
                         positive_embeddings: torch.Tensor,
                         negative_embeddings: torch.Tensor,
                         temperature: float = 0.1) -> torch.Tensor:
    """
    Compute InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    InfoNCE loss encourages the model to distinguish between positive and negative
    examples by maximizing agreement between anchor and positive while minimizing
    agreement with negatives.
    
    Args:
        anchor_embeddings: Anchor embeddings [batch_size, embed_dim]
        positive_embeddings: Positive embeddings [batch_size, embed_dim]  
        negative_embeddings: Negative embeddings [batch_size, embed_dim]
        temperature: Temperature parameter for softmax (lower = more selective)
        
    Returns:
        InfoNCE loss value
    """
    device = anchor_embeddings.device
    batch_size = anchor_embeddings.size(0)
    
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize embeddings for cosine similarity
    anchor_norm = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
    positive_norm = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
    negative_norm = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)
    
    # Compute similarities
    # Positive similarities: anchor-positive pairs
    pos_similarities = torch.sum(anchor_norm * positive_norm, dim=1) / temperature  # [batch_size]
    
    # Negative similarities: anchor-negative pairs
    neg_similarities = torch.sum(anchor_norm * negative_norm, dim=1) / temperature  # [batch_size]
    
    # For each anchor, we want to maximize similarity with its positive
    # and minimize similarity with its negative
    # InfoNCE: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
    
    # Numerically stable computation using logsumexp
    logits = torch.stack([pos_similarities, neg_similarities], dim=1)  # [batch_size, 2]
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)  # Positive is index 0
    
    # Cross-entropy loss where positive should have highest probability
    info_nce_loss = torch.nn.functional.cross_entropy(logits, targets)
    
    return info_nce_loss

def compute_contrastive_loss(student_logits: torch.Tensor,
                           labels: torch.Tensor,
                           tokenizer,
                           codebert_encoder,
                           triplet_sampler,
                           temperature: float = 0.1) -> torch.Tensor:
    """
    Compute contrastive loss using in-batch triplet sampling and InfoNCE.
    
    Args:
        student_logits: Student model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        tokenizer: Tokenizer for text conversion
        codebert_encoder: CodeBERT encoder for embeddings (may be cached)
        triplet_sampler: Triplet sampler for anchor/positive/negative
        temperature: Temperature for InfoNCE loss
        
    Returns:
        Contrastive loss value
    """
    device = student_logits.device
    
    try:
        # Sample triplets from batch
        triplets = triplet_sampler.sample_triplets(
            student_logits, labels, tokenizer
        )
        
        # Validate triplets
        if not triplet_sampler.validate_triplets(triplets):
            logger.warning("Triplet validation failed")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        num_triplets = len(triplets['anchor'])
        if num_triplets == 0:
            logger.warning("No triplets generated")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Encode triplets using CodeBERT (with caching if available)
        if hasattr(codebert_encoder, 'encode_code_pairs'):
            # Use optimized triplet encoding
            embeddings = codebert_encoder.encode_code_pairs(
                triplets['anchor'],
                triplets['positive'], 
                triplets['negative']
            )
        else:
            # Fallback to individual encoding
            all_texts = triplets['anchor'] + triplets['positive'] + triplets['negative']
            all_embeddings = codebert_encoder.encode_texts(all_texts)
            
            batch_size = len(triplets['anchor'])
            embeddings = {
                'anchor': all_embeddings[:batch_size],
                'positive': all_embeddings[batch_size:2*batch_size],
                'negative': all_embeddings[2*batch_size:]
            }
        
        # Move embeddings to correct device
        for key in embeddings:
            if embeddings[key].device != device:
                embeddings[key] = embeddings[key].to(device)
        
        # Compute InfoNCE loss
        contrastive_loss = compute_info_nce_loss(
            embeddings['anchor'],
            embeddings['positive'],
            embeddings['negative'],
            temperature=temperature
        )
        
        return contrastive_loss
        
    except Exception as e:
        # Graceful fallback if contrastive loss computation fails
        logger.warning(f"Failed to compute contrastive loss: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)

def ast_enhanced_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                     labels: torch.Tensor, tokenizer, ast_weight: float = 0.2, 
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    AST-enhanced distillation loss with syntax validation.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits  
        labels: Ground truth labels
        tokenizer: Tokenizer for text processing
        ast_weight: Weight for AST penalty
        **kwargs: Additional arguments (T, alpha, etc.)
        
    Returns:
        total_loss: Combined loss value
        component_losses: Dictionary of loss components
    """
    # Compute base distillation loss
    base_loss, base_components = optimized_distillation_loss_with_logging(
        student_logits, teacher_logits, labels, **kwargs
    )
    
    # Add AST penalty
    predictions = torch.argmax(student_logits, dim=-1)
    ast_penalty = compute_ast_penalty(predictions, tokenizer)
    
    # Combine losses
    total_loss = base_loss + ast_weight * ast_penalty
    
    # Update component dictionary
    component_losses = base_components.copy()
    component_losses['ast'] = ast_penalty.item() if hasattr(ast_penalty, 'item') else float(ast_penalty)
    component_losses['total'] = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
    
    return total_loss, component_losses


class RunningAverageNormalizer:
    """
    Running Average Normalizer for balancing loss component magnitudes.
    
    This class maintains exponential moving averages of loss components and normalizes
    them to ensure all components contribute similar gradient magnitudes during training.
    This addresses the issue where different loss functions have inherently different
    numerical scales (e.g., focal loss ~0.1-1.0 vs JSD loss ~10.0-12.0).
    
    The normalization formula is: normalized_loss = raw_loss / running_average_loss
    where running_average is updated as: avg = momentum * avg + (1-momentum) * raw_loss
    """
    
    def __init__(self, momentum: float = 0.99, min_scale: float = 1e-8, 
                 warmup_steps: int = 100, normalize_components: List[str] = None,
                 device: torch.device = None):
        """
        Initialize the running average normalizer.
        
        Args:
            momentum: Exponential moving average momentum (0.9-0.999)
            min_scale: Minimum running average to prevent division by zero
            warmup_steps: Number of steps before applying normalization
            normalize_components: List of loss component names to normalize
            device: PyTorch device for tensor operations
        """
        self.momentum = momentum
        self.min_scale = min_scale
        self.warmup_steps = warmup_steps
        self.device = device or torch.device('cpu')
        
        # Components to normalize
        self.normalize_components = normalize_components or [
            'focal', 'jsd', 'semantic', 'ce', 'kl', 'pans', 'ast', 'contrastive'
        ]
        
        # Running averages for each loss component
        self.running_averages = {}
        
        # Step counter for warmup
        self.step_count = 0
        
        # Initialize all components to None (will be set on first update)
        for component in self.normalize_components:
            self.running_averages[component] = None
    
    def update_and_normalize(self, loss_components: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update running averages and normalize loss components.
        
        Args:
            loss_components: Dictionary of raw loss values {component_name: loss_tensor}
            
        Returns:
            Dictionary of normalized loss values {component_name: normalized_loss_tensor}
        """
        self.step_count += 1
        normalized_components = {}
        
        for component_name, raw_loss in loss_components.items():
            if component_name not in self.normalize_components:
                # Don't normalize this component, pass through unchanged
                normalized_components[component_name] = raw_loss
                continue
            
            # Convert to scalar if tensor
            if isinstance(raw_loss, torch.Tensor):
                raw_value = raw_loss.item() if raw_loss.requires_grad else raw_loss.detach().item()
                device = raw_loss.device
            else:
                raw_value = float(raw_loss)
                device = self.device
            
            # Skip invalid values
            if torch.isnan(torch.tensor(raw_value)) or torch.isinf(torch.tensor(raw_value)):
                normalized_components[component_name] = raw_loss
                continue
            
            # Initialize running average on first encounter
            if self.running_averages[component_name] is None:
                self.running_averages[component_name] = abs(raw_value)
                normalized_components[component_name] = raw_loss
                continue
            
            # Update running average using exponential moving average
            current_avg = self.running_averages[component_name]
            new_avg = self.momentum * current_avg + (1 - self.momentum) * abs(raw_value)
            self.running_averages[component_name] = new_avg
            
            # Apply normalization after warmup period
            if self.step_count > self.warmup_steps and new_avg > self.min_scale:
                # Normalize: normalized_loss = raw_loss / running_average
                normalization_factor = max(new_avg, self.min_scale)
                
                if isinstance(raw_loss, torch.Tensor):
                    normalized_loss = raw_loss / normalization_factor
                else:
                    normalized_loss = torch.tensor(raw_value / normalization_factor, 
                                                 device=device, requires_grad=True)
                
                normalized_components[component_name] = normalized_loss
            else:
                # During warmup, pass through unchanged
                normalized_components[component_name] = raw_loss
        
        return normalized_components
    
    def get_normalization_stats(self) -> Dict[str, float]:
        """
        Get current normalization statistics for logging.
        
        Returns:
            Dictionary with running averages and normalization factors
        """
        stats = {
            'step_count': self.step_count,
            'warmup_complete': self.step_count > self.warmup_steps
        }
        
        for component, avg in self.running_averages.items():
            if avg is not None:
                stats[f'{component}_running_avg'] = avg
                stats[f'{component}_norm_factor'] = max(avg, self.min_scale)
        
        return stats
    
    def reset(self):
        """Reset all running averages and step count."""
        self.running_averages = {comp: None for comp in self.normalize_components}
        self.step_count = 0
    
    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            'running_averages': self.running_averages,
            'step_count': self.step_count,
            'momentum': self.momentum,
            'min_scale': self.min_scale,
            'warmup_steps': self.warmup_steps,
            'normalize_components': self.normalize_components
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dictionary from checkpoint."""
        self.running_averages = state_dict.get('running_averages', {})
        self.step_count = state_dict.get('step_count', 0)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.min_scale = state_dict.get('min_scale', self.min_scale)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.normalize_components = state_dict.get('normalize_components', self.normalize_components)
