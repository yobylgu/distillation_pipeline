"""
Multi-Compo    DEFAULT_WEIGHT_SCHEDULING = {
        'ce': {'start': 0.35, 'end': 0.25},
        'kl': {'start': 0.6, 'end': 0.35},
        'pans': {'start': 0.05, 'end': 0.25},
        'ast': {'start': 0.0, 'end': 0.15}
    }oss Architecture for advanced knowledge distillation.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple
from .loss_functions import compute_pans_loss, compute_ast_penalty, compute_focal_loss, compute_jsd_loss, compute_semantic_loss, compute_contrastive_loss, compute_weighted_cross_entropy

# Import defaults with fallback
try:
    from config.defaults import WEIGHT_SCHEDULING
except ImportError:
    # Fallback scheduling configuration (KL-prioritized aggressive preset)
    WEIGHT_SCHEDULING = {
        'ce': {'start': 0.35, 'end': 0.25},
        'kl': {'start': 0.6, 'end': 0.35},
        'pans': {'start': 0.05, 'end': 0.25},
        'ast': {'start': 0.0, 'end': 0.15}
    }

class MultiComponentLoss:
    """
    Multi-Component Loss Architecture for advanced knowledge distillation.
    
    Provides a framework for combining multiple loss components with dynamic weighting.
    This allows for sophisticated training strategies that can adapt based on training progress.
    """
    
    def __init__(self, components: List[str], weights: List[float], tokenizer=None, 
                 enable_dynamic_weighting: bool = True, custom_scheduling: Dict = None,
                 sentence_transformer_model=None, semantic_loss_scale: float = 5.0,
                 codebert_encoder=None, triplet_sampler=None, contrastive_temperature: float = 0.1,
                 token_weighter=None):
        """
        Initialize the multi-component loss.
        
        Args:
            components: List of loss component names ['ce', 'kl', 'pans', 'ast', 'focal', 'jsd', 'semantic', 'contrastive']
            weights: List of corresponding weights for each component (used as fallback)
            tokenizer: Tokenizer needed for text-based loss components
            enable_dynamic_weighting: Whether to enable dynamic weight scheduling
            custom_scheduling: Optional custom scheduling config, defaults to WEIGHT_SCHEDULING
            sentence_transformer_model: Pre-trained sentence transformer model for semantic loss
            semantic_loss_scale: β parameter for semantic loss scaling (scaled_sem = β × semantic_loss)
            codebert_encoder: CodeBERT encoder for contrastive learning
            triplet_sampler: Triplet sampler for contrastive learning
            contrastive_temperature: Temperature for InfoNCE loss
            token_weighter: NEW (Task 4.3) - CriticalTokenWeighter for per-token loss weighting
        """
        if len(components) != len(weights):
            raise ValueError("Components and weights must have same length")
            
        self.components = components
        self.tokenizer = tokenizer
        self.sentence_transformer_model = sentence_transformer_model
        self.semantic_loss_scale = semantic_loss_scale  # β parameter for semantic scaling
        self.codebert_encoder = codebert_encoder  # For contrastive learning
        self.triplet_sampler = triplet_sampler  # For contrastive learning
        self.contrastive_temperature = contrastive_temperature  # InfoNCE temperature
        self.token_weighter = token_weighter  # NEW: For token-specific weighting (Task 4.3)
        self.loss_history = {comp: [] for comp in components}
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.scheduling_config = custom_scheduling or WEIGHT_SCHEDULING
        
        # Initialize weights - use scheduled start values if dynamic weighting is enabled
        if enable_dynamic_weighting and self.scheduling_config:
            # Use scheduled start values as initial weights
            initial_weights = []
            for i, component in enumerate(components):
                if component in self.scheduling_config and 'start' in self.scheduling_config[component]:
                    initial_weights.append(self.scheduling_config[component]['start'])
                else:
                    # Fallback to provided weight
                    initial_weights.append(weights[i] if i < len(weights) else 0.1)
            self.initial_weights = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            # Use provided weights as-is when dynamic weighting is disabled
            self.initial_weights = torch.tensor(weights, dtype=torch.float32)
            
        self.weights = self.initial_weights.clone()
        
    def compute_component_loss(self, component: str, student_logits: torch.Tensor, 
                             teacher_logits: torch.Tensor, labels: torch.Tensor, 
                             T: float = 2.0, alpha: float = 0.5, **kwargs) -> torch.Tensor:
        """Compute individual loss component."""
        device = student_logits.device
        
        if component == 'ce':
            # Cross-entropy loss with optional token weighting (Task 4.3)
            token_weights = None
            if self.token_weighter is not None:
                token_weights = self.token_weighter.get_weight_tensor()
            return compute_weighted_cross_entropy(student_logits, labels, token_weights)
            
        elif component == 'kl':
            # Knowledge distillation loss
            active_mask = (labels != -100)
            if active_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
                
            student_flat = student_logits.reshape(-1, student_logits.size(-1))
            teacher_flat = teacher_logits.reshape(-1, teacher_logits.size(-1))
            active_flat = active_mask.reshape(-1)
            
            active_student = student_flat[active_flat]
            active_teacher = teacher_flat[active_flat]
            
            teacher_probs = F.softmax(active_teacher / T, dim=-1)
            student_log_probs = F.log_softmax(active_student / T, dim=-1)
            
            return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
            
        elif component == 'focal':
            # Focal Loss - replacement for cross-entropy with optional token weighting (Task 4.3)
            token_weights = None
            if self.token_weighter is not None:
                token_weights = self.token_weighter.get_weight_tensor()
            return compute_focal_loss(student_logits, labels, token_weights=token_weights)
            
        elif component == 'jsd':
            # Jensen-Shannon Divergence - replacement for KL divergence
            return compute_jsd_loss(student_logits, teacher_logits, T)
            
        elif component == 'semantic':
            # Semantic similarity loss with scaling (β parameter)
            if self.tokenizer is None or self.sentence_transformer_model is None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            raw_semantic_loss = compute_semantic_loss(student_logits, labels, self.tokenizer, self.sentence_transformer_model)
            # Apply semantic scaling: scaled_sem = β × semantic_loss
            scaled_semantic_loss = self.semantic_loss_scale * raw_semantic_loss
            return scaled_semantic_loss
            
        elif component == 'pans':
            # Position-Aware N-gram Similarity Loss
            if self.tokenizer is None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            return compute_pans_loss(student_logits, labels, self.tokenizer)
            
        elif component == 'ast':
            # AST-aware penalty
            if self.tokenizer is None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            predictions = torch.argmax(student_logits, dim=-1)
            return compute_ast_penalty(predictions, self.tokenizer)
            
        elif component == 'contrastive':
            # Contrastive learning loss (InfoNCE)
            if self.tokenizer is None or self.codebert_encoder is None or self.triplet_sampler is None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            return compute_contrastive_loss(
                student_logits, labels, self.tokenizer, 
                self.codebert_encoder, self.triplet_sampler,
                temperature=self.contrastive_temperature
            )
            
        else:
            raise ValueError(f"Unknown loss component: {component}")
            
        # Note: Backward compatibility is maintained - existing 'ce' and 'kl' components 
        # will continue to work alongside the new 'focal', 'jsd', and 'semantic' components
    
    def compute(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                labels: torch.Tensor, T: float = 2.0, alpha: float = 0.5, 
                step: int = None, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss as a weighted combination of all components.
        
        Args:
            step: Current training step for detailed logging
        
        Returns:
            total_loss: Combined loss value
            component_losses: Dictionary of individual loss values with enhanced logging
        """
        device = student_logits.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        component_losses = {}
        
        # Move weights to the same device
        weights = self.weights.to(device)
        
        # Enhanced logging: track raw scalars per mini-batch
        raw_component_scalars = {}
        weighted_component_scalars = {}
        
        for component, weight in zip(self.components, weights):
            try:
                loss = self.compute_component_loss(
                    component, student_logits, teacher_logits, labels, T, alpha, **kwargs
                )
                
                # Check for NaN/Inf in individual component loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf detected in {component} loss component: {loss}")
                    component_losses[component] = 0.0
                    raw_component_scalars[component] = 0.0
                    weighted_component_scalars[component] = 0.0
                    continue
                
                # Store raw scalar value (before weighting)
                raw_scalar = loss.item() if hasattr(loss, 'item') else float(loss)
                raw_component_scalars[component] = raw_scalar
                
                # Store weighted scalar value
                weighted_scalar = (weight * loss).item() if hasattr(weight * loss, 'item') else float(weight * loss)
                weighted_component_scalars[component] = weighted_scalar
                
                component_losses[component] = raw_scalar
                
                # Check component loss value after conversion
                if math.isnan(component_losses[component]) or math.isinf(component_losses[component]):
                    print(f"Warning: NaN/Inf in {component} after conversion: {component_losses[component]}")
                    component_losses[component] = 0.0
                    raw_component_scalars[component] = 0.0
                    weighted_component_scalars[component] = 0.0
                    continue
                
                total_loss = total_loss + weight * loss
                
                # Store in history for potential adaptive weighting
                self.loss_history[component].append(component_losses[component])
                
            except Exception as e:
                print(f"Warning: Error computing {component} loss: {e}")
                component_losses[component] = 0.0
                raw_component_scalars[component] = 0.0
                weighted_component_scalars[component] = 0.0
        
        # Calculate traditional combined loss for compatibility
        if 'ce' in component_losses and 'kl' in component_losses:
            traditional_total = alpha * component_losses['ce'] + (1 - alpha) * component_losses['kl']
            component_losses['traditional_total'] = traditional_total
        
        component_losses['total'] = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        
        # Enhanced logging metadata
        component_losses['_meta'] = {
            'step': step,
            'raw_scalars': raw_component_scalars,
            'weighted_scalars': weighted_component_scalars,
            'weights': {comp: weight.item() for comp, weight in zip(self.components, weights)},
            'semantic_loss_scale': self.semantic_loss_scale,  # Track β parameter
            'timestamp': torch.tensor(0.0).item()  # Placeholder for timestamp if needed
        }
        
        return total_loss, component_losses
    
    def update_weights(self, epoch: int, total_epochs: int):
        """
        Dynamically update component weights using linear scheduling strategy.
        
        Uses configurable start/end values from WEIGHT_SCHEDULING for smooth transitions.
        Linear interpolation formula: weight = start + progress * (end - start)
        where progress goes from 0.0 (start) to 1.0 (end of training).
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
        """
        if not self.enable_dynamic_weighting:
            return
            
        # Calculate training progress (0.0 to 1.0)
        # Use max(total_epochs - 1, 1) to avoid division by zero and ensure progress reaches 1.0
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        
        # Track old weights for logging
        old_weights = self.get_current_weights()
        
        # Update weights using linear interpolation
        updated_components = []
        for i, component in enumerate(self.components):
            if component in self.scheduling_config:
                start_weight = self.scheduling_config[component]['start']
                end_weight = self.scheduling_config[component]['end']
                
                # Linear interpolation: weight = start + progress * (end - start)
                new_weight = start_weight + progress * (end_weight - start_weight)
                
                # Apply weight bounds if configured
                try:
                    from config.defaults import WEIGHT_NORMALIZATION
                    if WEIGHT_NORMALIZATION.get('enabled', False):
                        min_weight = WEIGHT_NORMALIZATION.get('min_weight', 0.01)
                        max_weight = WEIGHT_NORMALIZATION.get('max_weight', 0.8)
                        new_weight = max(min_weight, min(max_weight, new_weight))
                except ImportError:
                    pass  # Skip bounds if config not available
                
                self.weights[i] = new_weight
                updated_components.append(f"{component}: {old_weights[component]:.3f} → {new_weight:.3f}")
        
        # Optional weight normalization (disabled by default for exact control)
        try:
            from config.defaults import WEIGHT_NORMALIZATION
            if WEIGHT_NORMALIZATION.get('enabled', False):
                total_weight = self.weights.sum()
                if total_weight > 0:
                    self.weights = self.weights / total_weight
        except ImportError:
            pass
        
        # Log significant weight changes (threshold: 0.01)
        new_weights = self.get_current_weights()
        significant_changes = []
        for comp in self.components:
            if comp in old_weights and abs(new_weights[comp] - old_weights[comp]) > 0.01:
                significant_changes.append(comp)
        
        if significant_changes:
            print(f"Epoch {epoch+1}: Weight updates for {significant_changes}")
            for update in updated_components:
                if any(comp in update for comp in significant_changes):
                    print(f"  {update}")
                    
        return new_weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current component weights as a dictionary."""
        return {comp: weight.item() for comp, weight in zip(self.components, self.weights)}
    
    def get_loss_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each loss component."""
        summary = {}
        for component, history in self.loss_history.items():
            if len(history) > 0:
                summary[component] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'latest': history[-1] if history else 0.0,
                    'trend': np.mean(np.diff(history[-10:])) if len(history) >= 10 else 0.0
                }
        return summary
    
    @classmethod
    def from_preset(cls, components: List[str], weights: List[float], tokenizer=None, 
                   preset_name: str = 'aggressive', enable_dynamic_weighting: bool = True):
        """
        Create MultiComponentLoss instance using the default aggressive scheduling.
        
        Args:
            components: List of loss component names
            weights: List of initial weights 
            tokenizer: Tokenizer for text-based losses
            preset_name: Preset name (only 'aggressive' supported, others ignored)
            enable_dynamic_weighting: Whether to enable dynamic scheduling
            
        Returns:
            MultiComponentLoss instance with aggressive scheduling configuration
        """
        # Always use the default WEIGHT_SCHEDULING (aggressive preset)
        return cls(components, weights, tokenizer, enable_dynamic_weighting, None)
    
    def apply_preset(self, preset_name: str):
        """
        Apply the default aggressive scheduling preset to this instance.
        
        Args:
            preset_name: Preset name (only 'aggressive' supported)
        """
        if preset_name == 'aggressive' or preset_name == 'default':
            self.scheduling_config = WEIGHT_SCHEDULING
            print(f"Applied aggressive scheduling preset")
        else:
            print(f"Warning: Only 'aggressive' preset is supported. Using default aggressive scheduling.")
