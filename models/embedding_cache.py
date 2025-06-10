"""
Embedding cache for minimizing computational overhead in contrastive learning.

Implements LRU cache for CodeBERT embeddings to avoid recomputing embeddings
for repeated text sequences during training.
"""

import hashlib
import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import threading
import time
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    LRU (Least Recently Used) cache for text embeddings.
    
    Caches embeddings to avoid recomputing for repeated text sequences.
    Thread-safe implementation with optional size limits and TTL (time-to-live).
    """
    
    def __init__(self, max_size: int = 10000, 
                 ttl_seconds: Optional[float] = None,
                 device: str = "cpu"):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            device: Device to store cached embeddings on
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.device = torch.device(device)
        
        # Thread-safe storage
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Initialized embedding cache: max_size={max_size}, ttl={ttl_seconds}s, device={device}")
    
    def _compute_hash(self, text: str) -> str:
        """Compute stable hash for text key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        age = time.time() - entry['timestamp']
        return age > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries (called while holding lock)."""
        if self.ttl_seconds is None:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry['timestamp'] > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1
    
    def _evict_lru(self):
        """Remove least recently used entry (called while holding lock)."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """
        Get cached embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding tensor or None if not found
        """
        key = self._compute_hash(text)
        
        with self._lock:
            # Clean expired entries
            self._evict_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._cache[key]
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                
                # Return embedding on correct device
                embedding = entry['embedding']
                if embedding.device != self.device:
                    embedding = embedding.to(self.device)
                    entry['embedding'] = embedding
                
                return embedding
            else:
                self._misses += 1
                return None
    
    def put(self, text: str, embedding: torch.Tensor):
        """
        Store embedding in cache.
        
        Args:
            text: Input text (key)
            embedding: Embedding tensor to cache
        """
        key = self._compute_hash(text)
        
        with self._lock:
            # Ensure embedding is on cache device
            if embedding.device != self.device:
                embedding = embedding.to(self.device)
            
            # Create cache entry
            entry = {
                'embedding': embedding.clone().detach(),  # Detach from computation graph
                'timestamp': time.time(),
                'text_length': len(text)
            }
            
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if necessary
            self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
    
    def get_batch(self, texts: List[str]) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get cached embeddings for batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (cached_embeddings, cache_miss_texts)
            cached_embeddings: List of embeddings (None for cache misses)
            cache_miss_texts: List of texts that need to be computed
        """
        cached_embeddings = []
        cache_miss_texts = []
        
        for text in texts:
            embedding = self.get(text)
            if embedding is not None:
                cached_embeddings.append(embedding)
                cache_miss_texts.append(None)  # Placeholder
            else:
                cached_embeddings.append(None)
                cache_miss_texts.append(text)
        
        # Filter out None values for cache misses
        actual_miss_texts = [text for text in cache_miss_texts if text is not None]
        
        return cached_embeddings, actual_miss_texts
    
    def put_batch(self, texts: List[str], embeddings: List[torch.Tensor]):
        """
        Store batch of embeddings in cache.
        
        Args:
            texts: List of input texts
            embeddings: List of corresponding embeddings
        """
        assert len(texts) == len(embeddings), "Texts and embeddings must have same length"
        
        for text, embedding in zip(texts, embeddings):
            if text is not None and embedding is not None:
                self.put(text, embedding)
    
    def clear(self):
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared embedding cache")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'total_requests': total_requests
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage of cached embeddings."""
        with self._lock:
            total_elements = 0
            total_bytes = 0
            
            for entry in self._cache.values():
                embedding = entry['embedding']
                elements = embedding.numel()
                bytes_per_element = embedding.element_size()
                
                total_elements += elements
                total_bytes += elements * bytes_per_element
            
            return {
                'total_embeddings': len(self._cache),
                'total_elements': total_elements,
                'total_bytes': total_bytes,
                'total_mb': total_bytes / (1024 * 1024),
                'avg_embedding_size': total_elements / len(self._cache) if len(self._cache) > 0 else 0
            }

class CachedCodeBERTEncoder:
    """
    CodeBERT encoder with embedding caching for improved performance.
    
    Wraps a CodeBERT encoder and adds caching functionality to minimize
    recomputation of embeddings for repeated text sequences.
    """
    
    def __init__(self, codebert_encoder, 
                 cache_size: int = 10000,
                 cache_ttl: Optional[float] = None,
                 cache_device: str = "cpu"):
        """
        Initialize cached encoder.
        
        Args:
            codebert_encoder: Base CodeBERT encoder
            cache_size: Maximum cache size
            cache_ttl: Cache TTL in seconds
            cache_device: Device for cached embeddings
        """
        self.encoder = codebert_encoder
        self.cache = EmbeddingCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
            device=cache_device
        )
        
        logger.info(f"Initialized cached CodeBERT encoder with cache size {cache_size}")
    
    def encode_texts(self, texts: List[str], 
                    max_length: int = 512, 
                    pooling: str = "cls") -> torch.Tensor:
        """
        Encode texts with caching.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            pooling: Pooling strategy
            
        Returns:
            Tensor of embeddings [batch_size, hidden_size]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache for each text
        cached_embeddings, cache_miss_texts = self.cache.get_batch(texts)
        
        # Compute embeddings for cache misses
        if cache_miss_texts:
            new_embeddings = self.encoder.encode_texts(
                cache_miss_texts, max_length=max_length, pooling=pooling
            )
            
            # Store new embeddings in cache
            self.cache.put_batch(cache_miss_texts, 
                               [new_embeddings[i] for i in range(len(cache_miss_texts))])
        else:
            new_embeddings = None
        
        # Combine cached and new embeddings
        result_embeddings = []
        new_idx = 0
        
        for i, text in enumerate(texts):
            if cached_embeddings[i] is not None:
                # Use cached embedding
                result_embeddings.append(cached_embeddings[i])
            else:
                # Use newly computed embedding
                result_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        # Stack into tensor
        return torch.stack(result_embeddings)
    
    def encode_code_pairs(self, anchor_texts: List[str], 
                         positive_texts: List[str],
                         negative_texts: List[str],
                         max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode triplets with caching.
        
        Args:
            anchor_texts: Anchor texts
            positive_texts: Positive texts
            negative_texts: Negative texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'anchor', 'positive', 'negative' embeddings
        """
        # Encode all texts together for better cache efficiency
        all_texts = anchor_texts + positive_texts + negative_texts
        all_embeddings = self.encode_texts(all_texts, max_length=max_length)
        
        # Split back into triplets
        batch_size = len(anchor_texts)
        return {
            'anchor': all_embeddings[:batch_size],
            'positive': all_embeddings[batch_size:2*batch_size],
            'negative': all_embeddings[2*batch_size:]
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get cache memory usage."""
        return self.cache.get_memory_usage()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()

def create_cached_encoder(codebert_encoder, 
                         cache_config: Optional[Dict] = None) -> CachedCodeBERTEncoder:
    """
    Factory function to create a cached CodeBERT encoder.
    
    Args:
        codebert_encoder: Base CodeBERT encoder
        cache_config: Optional cache configuration
        
    Returns:
        CachedCodeBERTEncoder instance
    """
    if cache_config is None:
        cache_config = {}
    
    return CachedCodeBERTEncoder(
        codebert_encoder,
        cache_size=cache_config.get('cache_size', 10000),
        cache_ttl=cache_config.get('cache_ttl', None),
        cache_device=cache_config.get('cache_device', 'cpu')
    )

# Example usage and testing
if __name__ == "__main__":
    # Test embedding cache
    print("Testing Embedding Cache...")
    
    cache = EmbeddingCache(max_size=5)
    
    # Test basic operations
    texts = ["hello world", "foo bar", "test code", "another text", "final text"]
    embeddings = [torch.randn(384) for _ in texts]
    
    # Put embeddings
    for text, emb in zip(texts, embeddings):
        cache.put(text, emb)
    
    print(f"Cache size: {cache.size()}")
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test retrieval
    for text in texts[:3]:
        cached_emb = cache.get(text)
        print(f"Retrieved embedding for '{text}': {cached_emb is not None}")
    
    # Test cache overflow
    cache.put("overflow text", torch.randn(384))
    print(f"Cache size after overflow: {cache.size()}")
    print(f"Final stats: {cache.get_stats()}")