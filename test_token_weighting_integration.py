#!/usr/bin/env python3
"""
Test script to verify token-specific weighting integration (Task 4.3).
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.loss_functions import compute_weighted_cross_entropy, compute_focal_loss
from models.multi_component_loss import MultiComponentLoss
from utils.token_weighting import CriticalTokenWeighter

class MockTokenizer:
    """Mock tokenizer for testing token weighting."""
    
    def __init__(self):
        # Create vocab with some critical tokens
        self.vocab = {
            'assertTrue': 100, 'assertFalse': 101, 'assertEquals': 102,
            'assertNull': 103, 'verify': 104, 'when': 105,
            'the': 1, 'and': 2, 'is': 3, 'of': 4, 'to': 5  # Non-critical tokens
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def get_vocab(self):
        return self.vocab
    
    def encode(self, text, add_special_tokens=False):
        words = text.lower().split()
        return [self.vocab.get(word, 0) for word in words]
    
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, (list, tuple)):
            return ' '.join([self.reverse_vocab.get(tid, '<unk>') for tid in token_ids])
        else:
            return self.reverse_vocab.get(token_ids, '<unk>')

def test_weighted_cross_entropy():
    """Test weighted cross-entropy loss with critical tokens."""
    print("🧪 Testing Weighted Cross-Entropy Loss")
    
    vocab_size = 110
    batch_size = 2
    seq_len = 4
    
    # Create sample data with critical and non-critical tokens
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([
        [100, 101, 1, 2],    # assertTrue, assertFalse, the, and
        [102, 103, 3, -100]  # assertEquals, assertNull, is, padding
    ])
    
    # Create token weights (critical tokens have weight 2.0, others 1.0)
    token_weights = torch.ones(vocab_size)
    critical_indices = [100, 101, 102, 103, 104, 105]  # assertion tokens
    for idx in critical_indices:
        token_weights[idx] = 2.0
    
    # Test unweighted loss
    loss_unweighted = compute_weighted_cross_entropy(logits, labels)
    
    # Test weighted loss
    loss_weighted = compute_weighted_cross_entropy(logits, labels, token_weights)
    
    print(f"Unweighted CE loss: {loss_unweighted.item():.4f}")
    print(f"Weighted CE loss: {loss_weighted.item():.4f}")
    
    # Weighted loss should be different (likely higher due to critical token emphasis)
    assert loss_unweighted.item() != loss_weighted.item(), "Weighted and unweighted losses should differ"
    
    print("✅ Weighted cross-entropy test passed")
    return True

def test_weighted_focal_loss():
    """Test focal loss with critical token weighting."""
    print("\n🧪 Testing Weighted Focal Loss")
    
    vocab_size = 110
    batch_size = 2
    seq_len = 4
    
    # Create sample data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([
        [100, 101, 1, 2],    # Critical tokens first
        [102, 103, 3, -100]  # Critical tokens first
    ])
    
    # Create token weights
    token_weights = torch.ones(vocab_size)
    critical_indices = [100, 101, 102, 103, 104, 105]
    for idx in critical_indices:
        token_weights[idx] = 3.0  # Higher weight for focal loss test
    
    # Test unweighted focal loss
    loss_unweighted = compute_focal_loss(logits, labels)
    
    # Test weighted focal loss
    loss_weighted = compute_focal_loss(logits, labels, token_weights=token_weights)
    
    print(f"Unweighted focal loss: {loss_unweighted.item():.4f}")
    print(f"Weighted focal loss: {loss_weighted.item():.4f}")
    
    # Weighted loss should be different
    assert loss_unweighted.item() != loss_weighted.item(), "Weighted and unweighted focal losses should differ"
    
    print("✅ Weighted focal loss test passed")
    return True

def test_multi_component_loss_integration():
    """Test integration with multi-component loss."""
    print("\n🧪 Testing Multi-Component Loss Integration")
    
    # Create tokenizer and weighter
    tokenizer = MockTokenizer()
    weighter = CriticalTokenWeighter(tokenizer, critical_token_weight=2.5)
    
    # Create multi-component loss with token weighting
    components = ['ce', 'focal']
    weights = [0.5, 0.5]
    
    multi_loss = MultiComponentLoss(
        components=components,
        weights=weights,
        tokenizer=tokenizer,
        enable_dynamic_weighting=False,
        token_weighter=weighter
    )
    
    # Create test data
    vocab_size = len(tokenizer.get_vocab())
    batch_size = 2
    seq_len = 3
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor([
        [100, 101, 1],  # assertTrue, assertFalse, the
        [102, 2, -100]  # assertEquals, and, padding
    ])
    
    # Compute loss with token weighting
    loss, loss_components = multi_loss.compute(student_logits, teacher_logits, labels)
    
    print(f"Multi-component loss: {loss.item():.4f}")
    print(f"Loss components: {loss_components}")
    
    # Verify loss was computed successfully
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert 'ce' in loss_components, "CE component should be present"
    assert 'focal' in loss_components, "Focal component should be present"
    
    print("✅ Multi-component loss integration test passed")
    return True

def test_token_weighter_integration():
    """Test CriticalTokenWeighter integration."""
    print("\n🧪 Testing CriticalTokenWeighter Integration")
    
    tokenizer = MockTokenizer()
    weighter = CriticalTokenWeighter(tokenizer, critical_token_weight=2.0)
    
    # Test weight tensor creation
    weight_tensor = weighter.get_weight_tensor()
    assert weight_tensor.shape[0] == len(tokenizer.get_vocab()), "Weight tensor should match vocab size"
    
    # Test critical token detection
    critical_indices = weighter.get_critical_indices()
    print(f"Critical indices found: {sorted(critical_indices)}")
    
    # Verify some tokens are marked as critical
    assert len(critical_indices) > 0, "Should find some critical tokens"
    
    # Test weight retrieval for specific tokens
    assertTrue_weight = weighter.get_token_weight(100)  # assertTrue
    the_weight = weighter.get_token_weight(1)  # the
    
    print(f"Weight for 'assertTrue' (token 100): {assertTrue_weight}")
    print(f"Weight for 'the' (token 1): {the_weight}")
    
    # Critical token should have higher weight
    assert assertTrue_weight > the_weight, "Critical token should have higher weight"
    
    print("✅ CriticalTokenWeighter integration test passed")
    return True

def main():
    """Run all token weighting tests."""
    print("🔑 Testing Token-Specific Weighting Integration (Task 4.3)")
    print("=" * 60)
    
    tests = [
        test_weighted_cross_entropy,
        test_weighted_focal_loss,
        test_multi_component_loss_integration,
        test_token_weighter_integration
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"📋 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All token weighting tests passed!")
        print("✅ Task 4.3 implementation verified")
    else:
        print("❌ Some tests failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)