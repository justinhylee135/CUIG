import torch
from typing import Dict, List, Optional

def uniform_merge(
    base_state_dict: Dict[str, torch.Tensor], 
    ckpt_state_dicts: List[Dict[str, torch.Tensor]], 
    key_filter: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Uniformly averages weights from multiple checkpoints.
    
    Args:
        base_state_dict: Base model state dictionary
        ckpt_state_dicts: List of checkpoint state dictionaries to merge
        key_filter: Only merge parameters whose keys contain this string
    
    Returns:
        Merged state dictionary
    """    
    # Initialize merged state with base model
    merged_state = base_state_dict.copy()
    merged_count = 0
    total_count = 0
    
    # Iterate through each parameter key
    for key in base_state_dict.keys():
        total_count += 1
        if key_filter is None or key_filter in key:
            merged_count += 1
            weights = []
            # Collect weights from all checkpoints for this key
            for sd in ckpt_state_dicts:
                if key not in sd:
                    raise ValueError(f"Key {key} not found in checkpoint")
                weights.append(sd[key])
            
            # Stack and compute mean
            avg_weight = torch.stack(weights, dim=0).mean(dim=0)
            merged_state[key] = avg_weight
    
    print(f"Merged {merged_count}/{total_count} parameters with filter '{key_filter}'")
    return merged_state