"""
Training utility functions for knowledge distillation.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

def get_dynamic_hyperparams(epoch, total_epochs, loss_history):
    """
    Dynamic temperature and alpha scheduling for improved knowledge distillation.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        loss_history: Dictionary containing loss history (e.g., {'ce': [], 'kl': [], 'total': []} 
                     or {'focal': [], 'jsd': [], 'total': []})
    
    Returns:
        tuple: (temperature, alpha)
    """
    # Exponential temperature decay: 4.0 → 1.5
    # Higher temperature initially for softer targets, lower towards end for sharper focus
    temperature = 4.0 * ((1.5/4.0) ** (epoch/total_epochs))
    
    # Adaptive alpha based on loss convergence
    # Find the main classification loss component (ce or focal)
    classification_component = None
    if 'ce' in loss_history and loss_history['ce']:
        classification_component = 'ce'
    elif 'focal' in loss_history and loss_history['focal']:
        classification_component = 'focal'
    
    if classification_component and len(loss_history[classification_component]) >= 3:
        # Check if classification loss is converging (small changes in recent epochs)
        class_trend = np.mean(np.diff(loss_history[classification_component][-3:]))
        if abs(class_trend) < 0.01:  # Classification loss converging, reduce alpha to focus more on distillation
            alpha = max(0.3, 0.5 - (epoch/total_epochs) * 0.2)
        else:
            alpha = 0.5  # Keep balanced if still changing
    else:
        alpha = 0.5  # Default balanced approach for early epochs or when no classification loss available
    
    return temperature, alpha

def setup_loss_function(args, tokenizer, sentence_transformer_model=None):
    """Setup loss function based on configuration."""
    loss_fn = None
    multi_loss = None
    
    if args.loss_function == 'multi_component':
        from models.multi_component_loss import MultiComponentLoss
        from config.defaults import WEIGHT_SCHEDULING, DEFAULT_LOSS_WEIGHTS
        
        components = args.loss_components
        enable_dynamic_weighting = getattr(args, 'enable_dynamic_weighting', True)
        
        # Use unified WEIGHT_SCHEDULING for all components (legacy + Trident)
        scheduling_config = WEIGHT_SCHEDULING
        
        if args.loss_weights is None:
            if enable_dynamic_weighting:
                # Use scheduled start values when dynamic weighting is enabled
                weights = []
                for comp in components:
                    if comp in scheduling_config and 'start' in scheduling_config[comp]:
                        weights.append(scheduling_config[comp]['start'])
                    else:
                        # Fallback to default weights for missing components
                        weights.append(DEFAULT_LOSS_WEIGHTS.get(comp, 0.1))
                print(f"Using unified dynamic weight scheduling - initial weights from WEIGHT_SCHEDULING")
            else:
                # Use static default weights when dynamic weighting is disabled
                weights = []
                for comp in components:
                    weights.append(DEFAULT_LOSS_WEIGHTS.get(comp, 0.1))
                print(f"Using static default weights from DEFAULT_LOSS_WEIGHTS")
        else:
            weights = args.loss_weights
            print(f"Using user-provided weights: {weights}")
            
        if len(weights) != len(components):
            raise ValueError(f"Number of weights ({len(weights)}) must match components ({len(components)})")
        
        # Initialize contrastive learning components if needed
        codebert_encoder = None
        triplet_sampler = None
        
        if 'contrastive' in components:
            from models.codebert_encoder import CodeBERTEncoder
            from models.triplet_sampler import InBatchTripletSampler
            
            print("Initializing CodeBERT encoder and triplet sampler for contrastive learning...")
            codebert_encoder = CodeBERTEncoder()
            triplet_sampler = InBatchTripletSampler()
        
        multi_loss = MultiComponentLoss(
            components, 
            weights, 
            tokenizer, 
            enable_dynamic_weighting=enable_dynamic_weighting,
            custom_scheduling=scheduling_config,
            sentence_transformer_model=sentence_transformer_model,
            codebert_encoder=codebert_encoder,
            triplet_sampler=triplet_sampler
        )
        
        if enable_dynamic_weighting:
            print(f"Using multi-component loss with unified dynamic weight scheduling")
            print(f"  Components: {components}, Initial scheduled weights: {weights}")
            expected_final = []
            for comp in components:
                if comp in scheduling_config and 'end' in scheduling_config[comp]:
                    expected_final.append(f"{comp}:{scheduling_config[comp]['end']:.2f}")
            print(f"  Will evolve to final weights: {expected_final}")
        else:
            print(f"Using multi-component loss with fixed weights")
            print(f"  Components: {components}, Static weights: {weights}")
    else:
        print(f"Using {args.loss_function} loss function")
    
    return loss_fn, multi_loss
