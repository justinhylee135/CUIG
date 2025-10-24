# Standard Library
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Third Party
import torch
import torch.nn as nn
from tqdm import tqdm


def accumulate_fisher(
    model: nn.Module,
    current_fisher: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Accumulate Fisher information during current unlearning.
    Won't be used until NEXT unlearning task.
    """    
    # Iterate through parameters with gradients
    for name, param in model.named_parameters():
        if param.grad is not None:

            # Estimate fisher with squared gradient
            grad_squared = param.grad.data.pow(2)
            
            # Store squared gradient
            if name in current_fisher:
                current_fisher[name] += grad_squared
            else:
                current_fisher[name] = grad_squared
    
    return current_fisher


def calculate_inverse_ewc_loss(
    model: nn.Module,
    fisher_dict: Dict[str, torch.Tensor],
    reference_params: Dict[str, torch.Tensor],
    device: torch.device,
    epsilon: float = 1e-8,
    temperature: float = 1.0,
    use_l2: bool = False,
    use_layerwise_norm: bool = False
) -> torch.Tensor:
    """
    Calculate Inverse EWC loss that encourages updates to previously important parameters.
    
    Args:
        model: Current model being trained
        fisher_dict: Dictionary of Fisher information values
        reference_params: Reference parameters (from previous unlearning step)
        device: Device for computation
        epsilon: Small value to prevent division by zero
        temperature: Temperature parameter for scaling
        use_l2: Whether to use L2 loss instead of L1

    Returns:
        Inverse EWC loss tensor
    """
    # Store original parameters for first iteration
    if not reference_params:
        print(f"Storing reference parameters for Inverse EWC regularization")
        for name, param in model.named_parameters():
            if param.requires_grad:
                reference_params[name] = param.detach().clone().requires_grad_(False)
            
        # Count total number of parameters elements
        total_elements = sum(p.numel() for p in reference_params.values())
        print(f"Stored '{len(reference_params)}' original parameters ('{total_elements:,}' total values) for Inverse EWC regularization")
        
    # Inverse EWC Loss
    loss = 0.0
    fisher_mean = None
    fisher_std = None
    
    if not use_layerwise_norm:
        all_fisher_values = []
        for name, fisher_tensor in fisher_dict.items():
            all_fisher_values.append(fisher_tensor.flatten())
        global_fisher = torch.cat(all_fisher_values)
        fisher_mean = global_fisher.mean().item()
        fisher_std = global_fisher.std().item() + epsilon
    
    # Iterate through parameters with fisher information and updated in previous unlearning run
    elements_included = 0
    for name, param in model.named_parameters():
        if name in fisher_dict and name in reference_params:
            
            # Keep track of parameter elements
            elements_included += param.numel()

            # Get Fisher information for this parameter
            fisher_value = fisher_dict[name].to(device)
            
            # Use layer-wise normalization if specified
            if use_layerwise_norm:
                fisher_mean = fisher_value.mean()
                fisher_std = fisher_value.std() + epsilon
                
            # Normalize fisher information
            fisher_normalized = (fisher_value - fisher_mean) / fisher_std

            # Apply inverse weighting - discourage changes where Fisher is low
            # Effectively encourages updates to parameters that were previously important
            inverse_weight = torch.exp(-fisher_normalized / temperature)

            # Compute parameter differenc
            if use_l2:
                # use L2 (Default for standard EWC)
                param_diff = (param - reference_params[name].to(device)).pow(2)
            else:
                # use L1
                param_diff = (param - reference_params[name].to(device)).abs()
            
            # Weight parameter difference by inverse weighting
            weighted_diff = inverse_weight * param_diff
            loss += weighted_diff.sum()
    # Normalize loss by number of elements included
    if elements_included > 0:
        loss /= elements_included

    return loss

def aggregate_fisher_dicts(
    current_fisher: Dict[str, torch.Tensor],
    previously_aggregated: Optional[Dict[str, torch.Tensor]],
    alpha: float = 0.70
) -> Dict[str, torch.Tensor]:
    """
    Aggregate this fisher dictionary with previous ones using EMA
    """

    # Return if this is the first fisher dictionary created
    if previously_aggregated is None:
        return current_fisher

    print(f"Aggregating fisher information with previous fisher using alpha: {alpha}")

    # Store new accumulation
    aggregate = {}

    # Iterate through fisher from this unlearning run
    for name in current_fisher:
        # Apply EMA
        if name in previously_aggregated:
            aggregate[name] = alpha * current_fisher[name] + (1 - alpha) * previously_aggregated[name]
        else:
            aggregate[name] = current_fisher[name]
    
    # Check for parameters that were not updated in this unlearning run but was previously updated
    for name in previously_aggregated:
        # Apply downscaling
        if name not in aggregate:
            aggregate[name] = (1-alpha) * previously_aggregated[name]
    
    # Return EMA Fisher dict
    return aggregate
    

def save_fisher_information(
    fisher_dict: Dict[str, torch.Tensor],
    save_path: str,
    iterations: int,
    previous_aggregated_fisher: Optional[Dict[str, torch.Tensor]] 
) -> None:
    """Save Fisher information dictionary to disk."""
    
    # Average accumulated fisher by number of iterations
    print(f"Fisher information averaged over '{iterations}' iterations")
    for name in fisher_dict:
        fisher_dict[name] /= iterations

    # Aggregate with previous Fisher
    if previous_aggregated_fisher is not None:
        fisher_dict = aggregate_fisher_dicts(fisher_dict, previous_aggregated_fisher)
    
    print(f"Saving fisher information to '{save_path}'")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher_dict, save_path)
    print(f"Saved Fisher information to '{save_path}'")


def load_fisher_information(
    load_path: str,
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """Load Fisher information dictionary from disk."""
    if os.path.exists(load_path):
        fisher_dict = torch.load(load_path, map_location=device)
        print(f"Loaded Fisher information from '{load_path}'")
        return fisher_dict
    else:
        print(f"No existing Fisher information found at '{load_path}'")
        return None