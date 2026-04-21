# Standard Library
import ast

# Third Party
import torch


def process_anchor(anchor_concepts, target_concepts):
    """Process anchor to ensure it matches the prompt list length."""
    # Anchor list can be a single anchor or a list of anchors
    if isinstance(anchor_concepts, list):
        anchor_list = anchor_concepts
    elif "[" in anchor_concepts:
        anchor_list = ast.literal_eval(anchor_concepts)
    else:
        anchor_list = [anchor_concepts]

    # Broadcast anchor list to match prompt list length
    if len(anchor_list) == 1:
        anchor_list = anchor_list * len(target_concepts)

    # If anchor still doesn't match prompt list length, raise an error
    if len(anchor_list) != len(target_concepts):
        raise ValueError(f"Anchor list length {len(anchor_list)} does not match prompt list length {len(target_concepts)}")

    return anchor_list


def build_global_topk_mask_dict(unet, grad_dict, topk):
    """Build a boolean mask dict by selecting the global top-k |param * grad| entries."""
    mask_dict = {}
    global_scores = []
    param_shapes = {}
    param_slices = {}

    total_params = 0
    total_elements = 0

    # Collect all |param*grad| values into a single vector
    for name, param in unet.named_parameters():
        # Only gradient activated parameters
        if not param.requires_grad:
            continue

        importance = (param.data * grad_dict[name]).abs().flatten()

        param_shapes[name] = param.shape
        param_slices[name] = (total_elements, total_elements + importance.numel())
        total_params += 1
        total_elements += importance.numel()
        global_scores.append(importance)

    # Build global top-k mask
    all_scores = torch.cat(global_scores)
    selected_elements = int(topk * total_elements)
    topk_idx = torch.topk(all_scores, selected_elements, largest=True).indices
    global_mask = torch.zeros_like(all_scores)
    global_mask[topk_idx] = 1

    # Identify which parameters have selected elements
    selected_params = 0
    for start, end in param_slices.values():
        flat_mask = global_mask[start:end]
        if flat_mask.any():  # if at least one element selected
            selected_params += 1

    # Output statistics
    print(f"Selected {selected_params} of {total_params} parameters or {(selected_params / total_params):.2%}")
    print(f"Selected {selected_elements:,} of {total_elements:,} elements or {(selected_elements / total_elements):.2%}")

    # Create final mask_dict
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            mask = torch.zeros_like(param)
        elif name in param_shapes:
            shape = param_shapes[name]
            start, end = param_slices[name]
            flat_mask = global_mask[start:end]
            mask = flat_mask.view(shape)
        else:
            # Not among selected group -> full zero mask
            mask = torch.zeros_like(param)

        mask_dict[name] = mask.bool()

    return mask_dict
