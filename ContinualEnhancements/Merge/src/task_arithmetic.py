import torch
from typing import Dict, List, Optional, Union

def task_arithmetic_merge(
    base_state_dict: Dict[str, torch.Tensor],
    ckpt_state_dicts: List[Dict[str, torch.Tensor]],
    key_filter: Optional[str] = None,
    lambda_val: Union[float, List[float]] = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Task arithmetic merging: averages task vectors (differences from base).
    
    Args:
        base_state_dict: Base model state dictionary
        ckpt_state_dicts: List of checkpoint state dictionaries to merge
        key_filter: Only merge parameters whose keys contain this string
        lambda_val: Scaling factor(s) for task vectors
    
    Returns:
        Merged state dictionary
    """
    # Load all checkpoints
    num_ckpts = len(ckpt_state_dicts)

    # Handle lambda scaling
    if not isinstance(lambda_val, list):
        lambda_vals = [lambda_val] * num_ckpts
    else:
        if len(lambda_val) != num_ckpts:
            raise ValueError("Length of lambda list must match number of checkpoints")
        lambda_vals = lambda_val
    
    merged_state = base_state_dict.copy()
    merged_count = 0
    total_count = 0
    
    print(f"Using lambda values:")
    for i, val in enumerate(lambda_vals):
        print(f"  Checkpoint {i+1}: {val}")
    
    # Iterate through base parameters
    for key, base_param in base_state_dict.items():
        total_count += 1
        if key_filter is None or key_filter in key:
            merged_count += 1
            base_vector = base_param.view(-1)
            total_task_vector = torch.zeros_like(base_vector)
            
            # Accumulate scaled task vectors
            for i, sd in enumerate(ckpt_state_dicts):
                if key not in sd:
                    raise ValueError(f"Key {key} not found in checkpoint")
                
                # Get task vector
                ckpt_param = sd[key].to(base_param.device)
                diff = (ckpt_param - base_param).view(-1)

                # Scale by lambda and add to total
                total_task_vector += lambda_vals[i] * diff
            
            # Merge back to base
            merged_state[key] = (base_vector + total_task_vector).view_as(base_param)
    
    print(f"Merged {merged_count}/{total_count} parameters with filter '{key_filter}'")
    return merged_state