"""
In-batch triplet sampling for contrastive learning in knowledge distillation.

Implements triplet sampling strategy where:
- Anchor: Ground truth (gold) sequence
- Positive: Student model prediction 
- Negative: Other sample's ground truth (gold) from the same batch
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)

class InBatchTripletSampler:
    """
    In-batch triplet sampler for contrastive learning.
    
    Creates triplets from batch data where each sample contributes:
    - Anchor: Its ground truth sequence
    - Positive: Student prediction for the same sample
    - Negative: Ground truth from a different sample in the batch
    """
    
    def __init__(self, negative_sampling_strategy: str = "random",
                 min_batch_size: int = 2,
                 max_negatives_per_anchor: int = 1):
        """
        Initialize triplet sampler.
        
        Args:
            negative_sampling_strategy: Strategy for selecting negatives
                - 'random': Random selection from other samples
                - 'hard': Select most similar negatives (future enhancement)
                - 'semi_hard': Select moderately similar negatives (future enhancement)
            min_batch_size: Minimum batch size required for triplet sampling
            max_negatives_per_anchor: Maximum number of negatives per anchor
        """
        self.negative_sampling_strategy = negative_sampling_strategy
        self.min_batch_size = min_batch_size
        self.max_negatives_per_anchor = max_negatives_per_anchor
        
        # Random state for reproducible sampling
        self.rng = np.random.RandomState(42)
        
        logger.info(f"Initialized triplet sampler with strategy: {negative_sampling_strategy}")
    
    def sample_triplets(self, student_logits: torch.Tensor,
                       labels: torch.Tensor,
                       tokenizer,
                       return_indices: bool = False) -> Dict[str, List[str]]:
        """
        Sample triplets from batch data.
        
        Args:
            student_logits: Student model predictions [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            tokenizer: Tokenizer for converting IDs to text
            return_indices: Whether to return triplet indices
            
        Returns:
            Dictionary with 'anchor', 'positive', 'negative' text lists
            Optionally includes 'indices' if return_indices=True
        """
        batch_size = student_logits.size(0)
        
        if batch_size < self.min_batch_size:
            logger.warning(f"Batch size {batch_size} < minimum {self.min_batch_size}, skipping triplet sampling")
            return {'anchor': [], 'positive': [], 'negative': []}
        
        # Get student predictions (argmax)
        student_predictions = torch.argmax(student_logits, dim=-1)  # [batch_size, seq_len]
        
        # Convert to text sequences
        anchor_texts = []  # Ground truth sequences
        positive_texts = []  # Student predictions
        triplet_indices = []
        
        for i in range(batch_size):
            # Extract anchor (ground truth)
            anchor_ids = labels[i]
            anchor_mask = (anchor_ids != -100)
            
            if anchor_mask.sum() == 0:
                continue  # Skip samples with no valid tokens
            
            anchor_tokens = anchor_ids[anchor_mask]
            
            # Extract positive (student prediction)
            positive_ids = student_predictions[i]
            positive_tokens = positive_ids[anchor_mask]  # Use same mask as anchor
            
            # Convert to text
            try:
                anchor_text = tokenizer.decode(anchor_tokens, skip_special_tokens=True)
                positive_text = tokenizer.decode(positive_tokens, skip_special_tokens=True)
                
                if len(anchor_text.strip()) > 0 and len(positive_text.strip()) > 0:
                    anchor_texts.append(anchor_text)
                    positive_texts.append(positive_text)
                    triplet_indices.append(i)
                    
            except Exception as e:
                logger.warning(f"Failed to decode sample {i}: {e}")
                continue
        
        # Sample negatives for each anchor
        negative_texts = self._sample_negatives(
            anchor_texts, triplet_indices, labels, tokenizer
        )
        
        # Ensure all lists have the same length
        min_length = min(len(anchor_texts), len(positive_texts), len(negative_texts))
        
        result = {
            'anchor': anchor_texts[:min_length],
            'positive': positive_texts[:min_length], 
            'negative': negative_texts[:min_length]
        }
        
        if return_indices:
            result['indices'] = triplet_indices[:min_length]
        
        logger.debug(f"Sampled {min_length} triplets from batch of {batch_size}")
        
        return result
    
    def _sample_negatives(self, anchor_texts: List[str],
                         anchor_indices: List[int],
                         labels: torch.Tensor,
                         tokenizer) -> List[str]:
        """
        Sample negative examples for each anchor.
        
        Args:
            anchor_texts: List of anchor texts
            anchor_indices: Indices of anchor samples in batch
            labels: Full batch labels tensor
            tokenizer: Tokenizer for conversion
            
        Returns:
            List of negative texts
        """
        negative_texts = []
        batch_size = labels.size(0)
        
        for i, anchor_idx in enumerate(anchor_indices):
            # Get candidate negative indices (all except current anchor)
            candidate_indices = [idx for idx in range(batch_size) if idx != anchor_idx]
            
            if len(candidate_indices) == 0:
                # Fallback: use anchor as negative (should not happen with min_batch_size > 1)
                negative_texts.append(anchor_texts[i])
                continue
            
            # Sample negative based on strategy
            if self.negative_sampling_strategy == "random":
                negative_idx = self.rng.choice(candidate_indices)
            else:
                # For future enhancement: implement hard/semi-hard sampling
                negative_idx = self.rng.choice(candidate_indices)
            
            # Extract negative text
            try:
                negative_ids = labels[negative_idx]
                negative_mask = (negative_ids != -100)
                
                if negative_mask.sum() == 0:
                    # Fallback if negative sample is empty
                    negative_texts.append(anchor_texts[i])
                    continue
                
                negative_tokens = negative_ids[negative_mask]
                negative_text = tokenizer.decode(negative_tokens, skip_special_tokens=True)
                
                if len(negative_text.strip()) == 0:
                    # Fallback if decoded text is empty
                    negative_texts.append(anchor_texts[i])
                else:
                    negative_texts.append(negative_text)
                    
            except Exception as e:
                logger.warning(f"Failed to sample negative for anchor {i}: {e}")
                negative_texts.append(anchor_texts[i])  # Fallback to anchor
        
        return negative_texts
    
    def sample_hard_negatives(self, anchor_embeddings: torch.Tensor,
                            candidate_embeddings: torch.Tensor,
                            candidate_texts: List[str],
                            num_negatives: int = 1) -> List[str]:
        """
        Sample hard negatives based on embedding similarity.
        
        Args:
            anchor_embeddings: Anchor embeddings [num_anchors, embed_dim]
            candidate_embeddings: Candidate negative embeddings [num_candidates, embed_dim]
            candidate_texts: Corresponding candidate texts
            num_negatives: Number of negatives to sample per anchor
            
        Returns:
            List of hard negative texts
        """
        # Compute cosine similarity
        anchor_norm = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
        candidate_norm = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
        similarity = torch.mm(anchor_norm, candidate_norm.t())  # [num_anchors, num_candidates]
        
        hard_negatives = []
        
        for i in range(similarity.size(0)):
            # Get similarity scores for this anchor
            sim_scores = similarity[i]
            
            # Sort by similarity (descending) to get hardest negatives
            # Hard negatives are those with high similarity but different meaning
            sorted_indices = torch.argsort(sim_scores, descending=True)
            
            # Take top negatives (excluding exact matches)
            selected_negatives = []
            for idx in sorted_indices:
                idx = idx.item()
                if len(selected_negatives) >= num_negatives:
                    break
                if sim_scores[idx] < 0.95:  # Avoid near-duplicates
                    selected_negatives.append(candidate_texts[idx])
            
            # Fallback to random if not enough hard negatives found
            while len(selected_negatives) < num_negatives:
                random_idx = self.rng.choice(len(candidate_texts))
                selected_negatives.append(candidate_texts[random_idx])
            
            hard_negatives.extend(selected_negatives)
        
        return hard_negatives
    
    def validate_triplets(self, triplets: Dict[str, List[str]]) -> bool:
        """
        Validate triplet data structure.
        
        Args:
            triplets: Triplet dictionary with 'anchor', 'positive', 'negative' keys
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['anchor', 'positive', 'negative']
        
        # Check all required keys present
        if not all(key in triplets for key in required_keys):
            logger.error(f"Missing required keys. Expected: {required_keys}, Got: {list(triplets.keys())}")
            return False
        
        # Check all lists have same length
        lengths = [len(triplets[key]) for key in required_keys]
        if not all(length == lengths[0] for length in lengths):
            logger.error(f"Inconsistent triplet lengths: {dict(zip(required_keys, lengths))}")
            return False
        
        # Check no empty lists (unless all are empty)
        if lengths[0] == 0:
            logger.warning("Empty triplet lists")
            return True
        
        # Check no empty strings
        for key in required_keys:
            for i, text in enumerate(triplets[key]):
                if not isinstance(text, str) or len(text.strip()) == 0:
                    logger.error(f"Empty or invalid text in {key}[{i}]: '{text}'")
                    return False
        
        logger.debug(f"Validated {lengths[0]} triplets successfully")
        return True
    
    def get_triplet_statistics(self, triplets: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute statistics for triplet quality assessment.
        
        Args:
            triplets: Triplet dictionary
            
        Returns:
            Dictionary with triplet statistics
        """
        if not self.validate_triplets(triplets):
            return {}
        
        num_triplets = len(triplets['anchor'])
        if num_triplets == 0:
            return {'num_triplets': 0}
        
        # Basic statistics
        stats = {
            'num_triplets': num_triplets,
            'avg_anchor_length': np.mean([len(text.split()) for text in triplets['anchor']]),
            'avg_positive_length': np.mean([len(text.split()) for text in triplets['positive']]),
            'avg_negative_length': np.mean([len(text.split()) for text in triplets['negative']]),
        }
        
        # Diversity statistics (approximate)
        anchor_unique = len(set(triplets['anchor']))
        positive_unique = len(set(triplets['positive']))
        negative_unique = len(set(triplets['negative']))
        
        stats.update({
            'anchor_diversity': anchor_unique / num_triplets,
            'positive_diversity': positive_unique / num_triplets, 
            'negative_diversity': negative_unique / num_triplets,
        })
        
        return stats

def create_triplet_sampler(config: Optional[Dict] = None) -> InBatchTripletSampler:
    """
    Factory function to create a triplet sampler with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured InBatchTripletSampler
    """
    if config is None:
        config = {}
    
    return InBatchTripletSampler(
        negative_sampling_strategy=config.get('negative_sampling_strategy', 'random'),
        min_batch_size=config.get('min_batch_size', 2),
        max_negatives_per_anchor=config.get('max_negatives_per_anchor', 1)
    )

# Example usage and testing
if __name__ == "__main__":
    # Test triplet sampler
    print("Testing In-Batch Triplet Sampler...")
    
    # Mock data
    batch_size, seq_len, vocab_size = 4, 10, 1000
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, -2:] = -100  # Add some padding
    
    # Mock tokenizer
    class MockTokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join([f"token_{t}" for t in tokens.tolist()])
    
    tokenizer = MockTokenizer()
    
    # Test sampler
    sampler = create_triplet_sampler()
    triplets = sampler.sample_triplets(student_logits, labels, tokenizer, return_indices=True)
    
    print(f"Sampled triplets: {len(triplets['anchor'])}")
    print(f"Validation: {sampler.validate_triplets(triplets)}")
    print(f"Statistics: {sampler.get_triplet_statistics(triplets)}")
    
    if len(triplets['anchor']) > 0:
        print(f"\nExample triplet:")
        print(f"Anchor: {triplets['anchor'][0]}")
        print(f"Positive: {triplets['positive'][0]}")
        print(f"Negative: {triplets['negative'][0]}")