"""
Token-specific weighting utilities for critical token loss enhancement.

This module handles mapping critical tokens to vocabulary indices and building
weight tensors for enhanced loss computation on important assertion tokens.

Task 4.2: Map critical tokens to vocab indices; build weight tensor; expose CLI 
param critical_token_weight (default 2).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path
import sys
import json

# Add config directory to path for critical tokens import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.critical_tokens import get_critical_tokens, get_token_categories, is_critical_token

logger = logging.getLogger(__name__)

class CriticalTokenWeighter:
    """
    Utility class for creating and managing token-specific weights for critical tokens.
    
    This class handles the mapping of critical assertion tokens to vocabulary indices
    and creates weight tensors that can be used to apply higher loss penalties to
    critical tokens during training.
    """
    
    def __init__(self, tokenizer, critical_token_weight: float = 2.0, 
                 device: str = "cpu", cache_file: Optional[str] = None):
        """
        Initialize the critical token weighter.
        
        Args:
            tokenizer: HuggingFace tokenizer with vocabulary
            critical_token_weight: Weight multiplier for critical tokens (default 2.0)
            device: Device to place weight tensors on
            cache_file: Optional path to cache token mappings
        """
        self.tokenizer = tokenizer
        self.critical_token_weight = critical_token_weight
        self.device = device
        self.cache_file = cache_file
        
        # Get critical tokens list
        self.critical_tokens = get_critical_tokens()
        
        # Initialize mappings
        self.vocab_size = len(tokenizer.get_vocab())
        self.critical_indices = set()
        self.token_to_indices = {}
        self.weight_tensor = None
        
        # Build mappings
        self._build_token_mappings()
        self._create_weight_tensor()
        
        logger.info(f"CriticalTokenWeighter initialized: {len(self.critical_indices)}/{self.vocab_size} tokens are critical")
    
    def _build_token_mappings(self):
        """Build mapping from critical tokens to vocabulary indices."""
        vocab = self.tokenizer.get_vocab()
        
        # Try to load from cache first
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.critical_indices = set(cache_data['critical_indices'])
                    self.token_to_indices = cache_data['token_to_indices']
                    logger.info(f"Loaded token mappings from cache: {self.cache_file}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding mappings")
        
        # Build mappings from scratch
        for token in self.critical_tokens:
            indices = self._find_token_indices(token, vocab)
            if indices:
                self.token_to_indices[token] = indices
                self.critical_indices.update(indices)
        
        # Save to cache if specified
        if self.cache_file:
            try:
                cache_data = {
                    'critical_indices': list(self.critical_indices),
                    'token_to_indices': self.token_to_indices,
                    'vocab_size': self.vocab_size,
                    'critical_token_weight': self.critical_token_weight
                }
                Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Saved token mappings to cache: {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def _find_token_indices(self, token: str, vocab: Dict[str, int]) -> List[int]:
        """
        Find vocabulary indices for a critical token using multiple strategies.
        
        Args:
            token: Critical token to find
            vocab: Tokenizer vocabulary mapping
            
        Returns:
            List of vocabulary indices that correspond to the token
        """
        indices = []
        
        # Strategy 1: Exact match
        if token in vocab:
            indices.append(vocab[token])
        
        # Strategy 2: Case variations
        variations = [token.lower(), token.upper(), token.capitalize()]
        for variation in variations:
            if variation in vocab and vocab[variation] not in indices:
                indices.append(vocab[variation])
        
        # Strategy 3: Subword tokenization (for tokens that might be split)
        try:
            # Tokenize the critical token to see how it's split
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            for token_id in token_ids:
                if token_id not in indices:
                    indices.append(token_id)
        except Exception:
            pass
        
        # Strategy 4: Context-based tokenization (token in typical assertion context)
        try:
            contexts = [
                f"{token}(",  # Method call context
                f".{token}",  # Method access context  
                f" {token} ",  # Standalone context
                f"@{token}",  # Annotation context
                f"{token};",  # Statement end context
            ]
            
            for context in contexts:
                try:
                    context_ids = self.tokenizer.encode(context, add_special_tokens=False)
                    # Look for the token within the context encoding
                    if len(context_ids) >= 1:
                        for token_id in context_ids:
                            if token_id not in indices:
                                # Verify this token actually represents our target
                                decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
                                if token.lower() in decoded.lower():
                                    indices.append(token_id)
                except Exception:
                    continue
        except Exception:
            pass
        
        return indices
    
    def _create_weight_tensor(self):
        """Create weight tensor with critical token weights."""
        # Initialize all weights to 1.0 (normal weight)
        weights = torch.ones(self.vocab_size, dtype=torch.float32)
        
        # Set critical token weights
        for idx in self.critical_indices:
            if idx < self.vocab_size:  # Safety check
                weights[idx] = self.critical_token_weight
        
        # Move to specified device
        self.weight_tensor = weights.to(self.device)
        
        logger.info(f"Created weight tensor: {len(self.critical_indices)} critical tokens with weight {self.critical_token_weight}")
    
    def get_weight_tensor(self) -> torch.Tensor:
        """
        Get the weight tensor for loss computation.
        
        Returns:
            Tensor of shape [vocab_size] with weights for each token
        """
        return self.weight_tensor
    
    def get_critical_indices(self) -> Set[int]:
        """
        Get set of critical token indices.
        
        Returns:
            Set of vocabulary indices that are critical tokens
        """
        return self.critical_indices.copy()
    
    def is_token_critical(self, token_id: int) -> bool:
        """
        Check if a token ID corresponds to a critical token.
        
        Args:
            token_id: Vocabulary index to check
            
        Returns:
            True if token is critical, False otherwise
        """
        return token_id in self.critical_indices
    
    def get_token_weight(self, token_id: int) -> float:
        """
        Get weight for a specific token ID.
        
        Args:
            token_id: Vocabulary index
            
        Returns:
            Weight for the token (1.0 for normal, critical_token_weight for critical)
        """
        if 0 <= token_id < self.vocab_size:
            return self.weight_tensor[token_id].item()
        return 1.0
    
    def analyze_critical_token_coverage(self, text_samples: List[str]) -> Dict:
        """
        Analyze how well critical tokens are covered in a sample of text.
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        total_tokens = 0
        critical_tokens_found = 0
        critical_token_counts = {}
        
        for text in text_samples:
            try:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(token_ids)
                
                for token_id in token_ids:
                    if self.is_token_critical(token_id):
                        critical_tokens_found += 1
                        decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        critical_token_counts[decoded_token] = critical_token_counts.get(decoded_token, 0) + 1
            except Exception:
                continue
        
        return {
            'total_tokens': total_tokens,
            'critical_tokens_found': critical_tokens_found,
            'critical_token_ratio': critical_tokens_found / total_tokens if total_tokens > 0 else 0.0,
            'unique_critical_tokens': len(critical_token_counts),
            'critical_token_distribution': critical_token_counts,
            'coverage_percentage': len(critical_token_counts) / len(self.critical_tokens) * 100
        }
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the critical token weighting setup.
        
        Returns:
            Dictionary with weighting statistics
        """
        return {
            'vocab_size': self.vocab_size,
            'total_critical_tokens_defined': len(self.critical_tokens),
            'critical_indices_mapped': len(self.critical_indices),
            'mapping_success_rate': len(self.critical_indices) / len(self.critical_tokens) * 100,
            'critical_token_weight': self.critical_token_weight,
            'device': str(self.device),
            'cache_file': self.cache_file
        }
    
    def update_weight(self, new_weight: float):
        """
        Update the critical token weight and rebuild tensor.
        
        Args:
            new_weight: New weight for critical tokens
        """
        self.critical_token_weight = new_weight
        self._create_weight_tensor()
        logger.info(f"Updated critical token weight to {new_weight}")

def create_critical_token_weighter(tokenizer, critical_token_weight: float = 2.0, 
                                 device: str = "cpu", cache_dir: Optional[str] = None) -> CriticalTokenWeighter:
    """
    Factory function to create a CriticalTokenWeighter.
    
    Args:
        tokenizer: HuggingFace tokenizer
        critical_token_weight: Weight multiplier for critical tokens
        device: Device for tensors
        cache_dir: Directory to store cache files
        
    Returns:
        CriticalTokenWeighter instance
    """
    cache_file = None
    if cache_dir:
        # Create cache filename based on tokenizer name
        model_name = getattr(tokenizer, 'name_or_path', 'unknown')
        safe_name = model_name.replace('/', '_').replace('\\', '_')
        cache_file = Path(cache_dir) / f"critical_tokens_cache_{safe_name}.json"
    
    return CriticalTokenWeighter(
        tokenizer=tokenizer,
        critical_token_weight=critical_token_weight,
        device=device,
        cache_file=str(cache_file) if cache_file else None
    )

# Example usage and testing
if __name__ == "__main__":
    print("🔑 Testing Critical Token Weighting System")
    
    # Mock tokenizer for testing (replace with real tokenizer in practice)
    class MockTokenizer:
        def __init__(self):
            # Create a small vocab with some critical tokens
            self.vocab = {
                'assertTrue': 100, 'assertFalse': 101, 'assertEquals': 102,
                'assertNull': 103, 'verify': 104, 'when': 105, 'then': 106,
                'assert': 107, 'True': 108, 'False': 109, 'null': 110,
                'test': 111, 'expected': 112, 'actual': 113,
                'the': 1, 'and': 2, 'is': 3, 'of': 4  # Common non-critical tokens
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        def get_vocab(self):
            return self.vocab
        
        def encode(self, text, add_special_tokens=False):
            # Simple word-based tokenization for testing
            words = text.lower().split()
            return [self.vocab.get(word, 0) for word in words]
        
        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, (list, tuple)):
                return ' '.join([self.reverse_vocab.get(tid, '<unk>') for tid in token_ids])
            else:
                return self.reverse_vocab.get(token_ids, '<unk>')
    
    # Test the weighter
    tokenizer = MockTokenizer()
    weighter = CriticalTokenWeighter(tokenizer, critical_token_weight=2.5)
    
    print(f"📊 Statistics: {weighter.get_statistics()}")
    print(f"🎯 Critical indices: {sorted(weighter.get_critical_indices())}")
    print(f"⚖️  Weight tensor shape: {weighter.get_weight_tensor().shape}")
    
    # Test with sample assertions
    test_assertions = [
        "assertTrue expected actual",
        "assertEquals the result",
        "verify when then test"
    ]
    
    coverage = weighter.analyze_critical_token_coverage(test_assertions)
    print(f"📈 Coverage analysis: {coverage}")
    
    print("✅ Critical token weighting system ready for Task 4.3")