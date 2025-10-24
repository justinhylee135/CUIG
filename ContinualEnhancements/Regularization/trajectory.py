# Standard Library
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Third Party
import torch
import torch.nn as nn
from tqdm import tqdm

def calculate_trajectory_loss(
    model: nn.Module,
    previous_changes: Dict[str, torch.Tensor],
    reference_params: Dict[str, torch.Tensor],
    device: torch.device,
    epsilon: float = 1e-8,
    temperature: float = 1.0,
    use_l2: bool = False,
    use_layerwise_norm: bool = False
) -> torch.Tensor:
    """
    Calculate trajectory-based regularization loss using parameter change history.
    
    Args:
        model: Current model being trained
        previous_changes: Dictionary of parameter changes from previous step
        reference_params: Reference parameters for current loss calculation
        device: Device for computation
        epsilon: Small value to prevent division by zero
        temperature: Temperature parameter for scaling
        use_l2: Whether to use L2 loss instead of L1
        use_layerwise_norm: Whether to use layer-wise normalization

    Returns:
        Trajectory regularization loss tensor
    """
    # Store original parameters for first iteration
    if not reference_params:
        print(f"\tStoring reference parameters for trajectory regularization")
        for name, param in model.named_parameters():
            if param.requires_grad:
                reference_params[name] = param.detach().clone().requires_grad_(False)
            
        # Count total number of parameters elements
        total_elements = sum(p.numel() for p in reference_params.values())
        print(f"\tStored '{len(reference_params)}' original parameters ('{total_elements:,}' total values) for trajectory regularization")
        
    # If no previous changes available, return zero loss
    if not previous_changes:
        return torch.tensor(0.0, device=device)
        
    # Trajectory regularization loss
    loss = 0.0
    change_mean = None
    change_std = None
    
    # Global normalization if specified
    if not use_layerwise_norm:
        all_change_values = []
        for name, change_tensor in previous_changes.items():
            all_change_values.append(change_tensor.flatten())
        
        if all_change_values:
            global_changes = torch.cat(all_change_values)
            change_mean = global_changes.mean().item()
            change_std = global_changes.std().item() + epsilon
    
    # Iterate through parameters with change history
    elements_included = 0
    for name, param in model.named_parameters():
        if name in previous_changes and name in reference_params:
            
            # Keep track of parameter elements
            elements_included += param.numel()

            # Get parameter changes from previous step
            prev_change = previous_changes[name].to(device)
            
            # Use layer-wise normalization if specified
            if use_layerwise_norm:
                change_mean = prev_change.mean()
                change_std = prev_change.std() + epsilon
                
            # Normalize parameter changes
            change_normalized = (prev_change - change_mean) / change_std

            # Discourage updates to parameters that didn't change much
            weight = torch.exp(-change_normalized / temperature)

            # Compute current parameter difference
            if use_l2:
                # Use L2 (squared differences)
                param_diff = (param - reference_params[name].to(device)).pow(2)
            else:
                # Use L1 (absolute differences)
                param_diff = (param - reference_params[name].to(device)).abs()
            
            # Weight parameter difference by change history
            weighted_diff = weight * param_diff
            loss += weighted_diff.sum()
    
    # Normalize loss by number of elements included
    if elements_included > 0:
        loss /= elements_included

    return loss


def aggregate_parameter_changes(
    current_changes: Dict[str, torch.Tensor],
    previously_aggregated: Optional[Dict[str, torch.Tensor]],
    alpha: float = 0.70
) -> Dict[str, torch.Tensor]:
    """
    Aggregate parameter changes with previous ones using exponential moving average (EMA).
    
    Args:
        current_changes: Parameter changes from current unlearning step
        previously_aggregated: Previously aggregated changes
        alpha: EMA coefficient (higher = more weight to current changes)
    
    Returns:
        Aggregated parameter changes dictionary
    """

    # Return if this is the first change dictionary created
    if previously_aggregated is None:
        return current_changes

    print(f"Aggregating parameter changes with previous changes using alpha: {alpha}")

    # Store new accumulation
    aggregate = {}

    # Iterate through changes from this unlearning run
    for name in current_changes:
        # Apply EMA
        if name in previously_aggregated:
            aggregate[name] = alpha * current_changes[name] + (1 - alpha) * previously_aggregated[name]
        else:
            aggregate[name] = current_changes[name]
    
    # Check for parameters that were not updated in this unlearning run but were previously updated
    for name in previously_aggregated:
        # Apply downscaling to maintain memory of previous changes
        if name not in aggregate:
            aggregate[name] = (1 - alpha) * previously_aggregated[name]
    
    # Return EMA parameter changes dict
    return aggregate
    

def save_delta_to_path(
    model: torch.nn.Module,
    original_params: Dict[str, torch.Tensor],
    save_path: str,
    previous_aggregated_delta: Optional[Dict[str, torch.Tensor]] 
) -> None:
    """
    Save parameter changes dictionary to disk.
    
    Args:
       model: The model whose parameters have changed
       original_params: The original parameters of the model
       save_path: Path to save the changes
       previous_aggregated_delta: Previously aggregated changes for EMA
    """

    delta_dict = {}
    # Compute parameter deltas
    for name, param in model.named_parameters():
        if name in original_params:
            delta_dict[name] = torch.abs(param.data - original_params[name])

    # Aggregate with previous changes
    if previous_aggregated_delta is not None:
        delta_dict = aggregate_parameter_changes(delta_dict, previous_aggregated_delta)
    
    print(f"Saving parameter changes to '{save_path}'")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta_dict, save_path)
    print(f"Saved parameter changes to '{save_path}'")


def load_delta_from_path(
    load_path: str,
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load parameter changes dictionary from disk.
    
    Args:
        load_path: Path to load changes from
        device: Device to load tensors to
    
    Returns:
        Parameter changes dictionary or None if not found
    """
    if os.path.exists(load_path):
        changes_dict = torch.load(load_path, map_location=device)
        print(f"\tLoaded parameter changes from '{load_path}'")
        return changes_dict
    else:
        print(f"\tNo existing parameter changes found at '{load_path}'")
        return None