"""
CodeBERT encoder module for contrastive learning in knowledge distillation.

This module provides a frozen CodeBERT encoder for computing embeddings
used in contrastive semantic loss (InfoNCE).
"""

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import List, Union, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CodeBERTEncoder(nn.Module):
    """
    Frozen CodeBERT encoder for contrastive learning.
    
    Uses microsoft/codebert-base as a frozen feature extractor for computing
    code embeddings used in triplet sampling and InfoNCE loss computation.
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", 
                 cache_dir: Optional[str] = None, device: str = "auto"):
        """
        Initialize CodeBERT encoder.
        
        Args:
            model_name: HuggingFace model identifier for CodeBERT
            cache_dir: Optional cache directory for model files
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        logger.info(f"Loading CodeBERT encoder: {model_name}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.model = RobertaModel.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            
            # Move to device and freeze parameters
            self.model.to(self.device)
            self._freeze_parameters()
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"CodeBERT encoder loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CodeBERT encoder: {e}")
            raise
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _freeze_parameters(self):
        """Freeze all model parameters to prevent training."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("CodeBERT parameters frozen")
    
    def encode_texts(self, texts: Union[str, List[str]], 
                    max_length: int = 512, 
                    pooling: str = "cls") -> torch.Tensor:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            max_length: Maximum sequence length for tokenization
            pooling: Pooling strategy ('cls', 'mean', 'max')
            
        Returns:
            Tensor of embeddings [batch_size, hidden_size]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # Apply pooling strategy
            if pooling == "cls":
                # Use [CLS] token embedding
                embeddings = hidden_states[:, 0, :]
            elif pooling == "mean":
                # Mean pooling over sequence (excluding padding)
                attention_mask = inputs["attention_mask"]
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif pooling == "max":
                # Max pooling over sequence
                embeddings = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embeddings
    
    def encode_code_pairs(self, anchor_texts: List[str], 
                         positive_texts: List[str],
                         negative_texts: List[str],
                         max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode triplets of texts for contrastive learning.
        
        Args:
            anchor_texts: List of anchor (reference) texts
            positive_texts: List of positive (similar) texts  
            negative_texts: List of negative (dissimilar) texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'anchor', 'positive', 'negative' embeddings
        """
        # Ensure all lists have the same length
        batch_size = len(anchor_texts)
        assert len(positive_texts) == batch_size
        assert len(negative_texts) == batch_size
        
        # Encode all texts together for efficiency
        all_texts = anchor_texts + positive_texts + negative_texts
        all_embeddings = self.encode_texts(all_texts, max_length=max_length)
        
        # Split embeddings back into triplets
        anchor_embeddings = all_embeddings[:batch_size]
        positive_embeddings = all_embeddings[batch_size:2*batch_size]  
        negative_embeddings = all_embeddings[2*batch_size:]
        
        return {
            'anchor': anchor_embeddings,
            'positive': positive_embeddings,
            'negative': negative_embeddings
        }
    
    def compute_similarity_matrix(self, embeddings1: torch.Tensor, 
                                embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch1, hidden_size]
            embeddings2: Second set of embeddings [batch2, hidden_size]
            
        Returns:
            Similarity matrix [batch1, batch2]
        """
        # Normalize embeddings
        norm1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        norm2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(norm1, norm2.t())
        
        return similarity
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.model.config.hidden_size
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                pooling: str = "cls") -> torch.Tensor:
        """
        Forward pass for integration with PyTorch modules.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pooling: Pooling strategy
            
        Returns:
            Embeddings [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            if pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            elif pooling == "mean":
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif pooling == "max":
                embeddings = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embeddings

def load_codebert_encoder(model_name: str = "microsoft/codebert-base",
                         device: str = "auto",
                         cache_dir: Optional[str] = None) -> CodeBERTEncoder:
    """
    Convenience function to load a CodeBERT encoder.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run on
        cache_dir: Optional cache directory
        
    Returns:
        Initialized CodeBERTEncoder
    """
    return CodeBERTEncoder(model_name=model_name, device=device, cache_dir=cache_dir)

# Example usage
if __name__ == "__main__":
    # Test the CodeBERT encoder
    encoder = load_codebert_encoder()
    
    # Test with sample code
    sample_code = [
        "public void testEquals() { assertEquals(expected, actual); }",
        "private String getName() { return this.name; }",
        "assertTrue(list.isEmpty());"
    ]
    
    embeddings = encoder.encode_texts(sample_code)
    print(f"Encoded {len(sample_code)} code samples")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")