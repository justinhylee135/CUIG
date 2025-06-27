import torch
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import os

def ties_merge(
    base_state_dict: Dict[str, torch.Tensor],
    ckpt_state_dicts: List[Dict[str, torch.Tensor]],
    key_filter: Optional[str] = None,
    lambda_val: Union[float, List[float]] = 1.0,
    top_k: Union[float, List[float]] = 0.3,
    merge_func: str = "mean"
) -> Dict[str, torch.Tensor]:
    """
    Perform TIES merging on model checkpoints.
    
    Args:
        base_state_dict: Base model state dictionary
        ckpt_state_dicts: List of checkpoint state dictionaries to merge
        key_filter: Only merge parameters whose keys contain this string
        lambda_val: Scaling factor(s) for task vectors
        top_k: Top-K fraction(s) for sparsification
        merge_func: Merging function ("mean", "sum", or "max")
    
    Returns:
        Merged state dictionary
    """
    num_ckpts = len(ckpt_state_dicts)

    # Handle lambda scaling
    if not isinstance(lambda_val, list):
        lambda_vals = [lambda_val] * num_ckpts
        if lambda_val != 1.0:
            merge_func = "mean"  # Use mean when global lambda is specified
    else:
        if len(lambda_val) != num_ckpts:
            raise ValueError("Length of lambda list must match number of checkpoints")
        lambda_vals = lambda_val
    
    # Handle top-K values
    if not isinstance(top_k, list):
        top_k_vals = [top_k] * num_ckpts
    else:
        if len(top_k) != num_ckpts:
            raise ValueError("Length of top_k list must match number of checkpoints")
        top_k_vals = top_k
    
    for i, val in enumerate(lambda_vals):
        print(f"Checkpoint {i+1}: lambda={val:.3f}, top_k={top_k_vals[i]:.3f}")
    
    # Initialize merged state dictionary
    merged_state = base_state_dict.copy()
    merged_count = 0
    total_count = 0
    progress_bar = tqdm(base_state_dict.items(), desc="TIES Merging")
    
    # Iterate through base parameters
    for key, base_param in progress_bar:
        total_count += 1
        
        if key_filter is None or key_filter in key:
            merged_count += 1
            base_vector = base_param.view(-1)
            task_vectors = []
            
            # Compute task vectors (differences from base)
            for i, state_dict in enumerate(ckpt_state_dicts):
                if key not in state_dict:
                    raise ValueError(f"Key {key} not found in checkpoint {i}")
                
                # Task Vector
                param = state_dict[key].to(base_param.device)
                diff = (param - base_param).view(-1)

                
                # Scale by lambda
                diff = lambda_vals[i] * diff
                task_vectors.append(diff)
            
            # Stack task vectors for TIES processing
            task_matrix = torch.stack(task_vectors, dim=0)
            
            # Apply TIES merging
            ties_vector = apply_ties(
                task_matrix, 
                top_k_vals, 
                merge_func, 
                verbose=False
            )
            
            # Add to base parameter
            final_vector = base_vector + ties_vector
            merged_state[key] = final_vector.view_as(base_param)
    
    print(f"Merged {merged_count}/{total_count} parameters with filter '{key_filter}' and '{merge_func}' merge function")
    
    return merged_state

def apply_ties(
    task_vector_matrix: torch.Tensor, 
    top_k: Union[float, List[float]], 
    merge_func: str = "mean", 
    verbose: bool = False
) -> torch.Tensor:
    """
    Core TIES merging algorithm.
    
    Args:
        task_vector_matrix: Matrix of task vectors (num_tasks x num_params)
        top_k: Top-K fraction(s) to keep per task
        merge_func: Merging function ("mean", "sum", or "max")
        verbose: Print debug information
    
    Returns:
        Merged vector
    """
    # Clone input to avoid modifying original
    cloned_tv = task_vector_matrix.clone()
    
    # Apply top-k masking to each row (task)
    if verbose:
        print("Applying top-K masking...")
    masked_tv, _ = topk_values_mask(cloned_tv, top_k, return_mask=False)
    
    # Resolve sign conflicts
    if verbose:
        print("Resolving sign conflicts...")
    dominant_signs = resolve_sign(masked_tv)
    
    # Perform disjoint merging
    if verbose:
        print(f"Performing disjoint merge with function: {merge_func}")
    ties_vector = disjoint_merge(masked_tv, merge_func, dominant_signs)
    
    return ties_vector


def topk_values_mask(
    M: torch.Tensor, 
    top_k: Union[float, List[float]], 
    return_mask: bool = False
):
    """
    Apply top-K masking to each row of matrix M.
    
    Args:
        M: Input matrix (num_tasks x num_params)
        top_k: Top-K fraction(s) to keep
        return_mask: Whether to return the mask
    
    Returns:
        Masked matrix and optionally the mask
    """
    if M.dim() == 1:
        M = M.unsqueeze(0)
    
    n, d = M.shape
    masks = []
    
    for i in range(n):
        # Get top-K value for this row
        if isinstance(top_k, (list, tuple)):
            current_top_k = top_k[i]
        else:
            current_top_k = top_k
        
        # Convert percentage to fraction if needed
        if current_top_k > 1:
            current_top_k = current_top_k / 100.0
        
        # Calculate number of values to keep
        k = int(d * current_top_k)
        k_drop = d - k
        
        if k_drop > 0:
            # Find threshold value
            bottom_value = M[i].abs().kthvalue(k_drop, keepdim=True).values
            mask = M[i].abs() >= bottom_value
        else:
            # Keep all values
            mask = torch.ones_like(M[i], dtype=torch.bool)
        
        masks.append(mask)
    
    # Stack masks
    final_mask = torch.stack(masks, dim=0)
    masked_M = M * final_mask
    
    if return_mask:
        return masked_M, final_mask.float().mean(dim=1), final_mask
    
    return masked_M, final_mask.float().mean(dim=1)


def resolve_sign(masked_tv: torch.Tensor) -> torch.Tensor:
    """
    Resolve sign conflicts by finding dominant sign for each parameter.
    
    Args:
        masked_tv: Masked task vector matrix
    
    Returns:
        Dominant sign tensor
    """
    # Sum across tasks to get dominant sign
    dominant_sign = torch.sign(masked_tv.sum(dim=0))
    
    # Resolve zeros using majority vote
    dominant_sign = resolve_zero_signs(dominant_sign, method="majority")
    
    return dominant_sign


def resolve_zero_signs(dominant_sign: torch.Tensor, method: str = "majority") -> torch.Tensor:
    """
    Resolve zero signs in dominant sign tensor.
    
    Args:
        dominant_sign: Tensor with signs (-1, 0, 1)
        method: "majority" or "minority"
    
    Returns:
        Resolved sign tensor
    """
    # Get overall majority sign
    majority_sign = torch.sign(dominant_sign.sum())
    
    if method == "majority":
        # Set zeros to majority sign
        dominant_sign[dominant_sign == 0] = majority_sign
    elif method == "minority":
        # Set zeros to minority sign
        dominant_sign[dominant_sign == 0] = -majority_sign
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dominant_sign


def disjoint_merge(
    masked_tv: torch.Tensor, 
    merge_func: str, 
    dominant_sign: torch.Tensor
) -> torch.Tensor:
    """
    Perform disjoint merging of task vectors.
    
    Args:
        masked_tv: Masked task vector matrix
        merge_func: Merging function ("mean", "sum", or "max")
        dominant_sign: Dominant sign for each parameter
    
    Returns:
        Merged vector
    """
    # Remove any prefix from merge function name
    merge_func = merge_func.split("-")[-1]
    
    # Align task vectors with dominant signs
    if dominant_sign is not None:
        # Keep only values that align with dominant sign
        alignment_mask = torch.where(
            dominant_sign.unsqueeze(0) > 0, 
            masked_tv > 0, 
            masked_tv < 0
        )
        aligned_tv = masked_tv * alignment_mask
    else:
        # Keep all non-zero values
        alignment_mask = masked_tv != 0
        aligned_tv = masked_tv * alignment_mask
    
    # Perform merging
    if merge_func == "mean":
        # Average non-zero aligned values
        count_nonzero = (aligned_tv != 0).sum(dim=0).float()
        merged_vector = aligned_tv.sum(dim=0) / torch.clamp(count_nonzero, min=1)
        
    elif merge_func == "sum":
        # Sum all aligned values
        merged_vector = aligned_tv.sum(dim=0)
        
    elif merge_func == "max":
        # Take maximum magnitude with dominant sign
        merged_vector = aligned_tv.abs().max(dim=0)[0]
        merged_vector *= dominant_sign
        
    else:
        raise ValueError(f"Unknown merge function: {merge_func}")
    
    return merged_vector