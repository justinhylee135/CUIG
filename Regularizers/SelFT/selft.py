# Standard Library
import os

# Third Party
import torch

# Local
from .src.helpers import build_global_topk_mask_dict
from .src.scoring import selft_get_score

def get_selft_mask_dict(unet, text_encoder, tokenizer, mask_dict_path, grad_dict_path, target_concepts, anchor_concepts, topk, loss, device, diffusion_pipe=None):
    """Get the SelFT mask for the UNet model"""  
    # Check if the mask dictionary exists
    if mask_dict_path is not None and os.path.exists(mask_dict_path):
        print(f"\nLoading SelFT mask dictionary from '{mask_dict_path}'...")
        mask_dict = torch.load(mask_dict_path, map_location=device)
        print(f"\nLoaded SelFT mask dictionary from '{mask_dict_path}'")
        print(f"Skipping SelFT mask calculation")
        return mask_dict
    
    # Check if the grad dictionary exists
    if grad_dict_path is not None and os.path.exists(grad_dict_path):
        grad_dict = torch.load(grad_dict_path, map_location=device)
        print(f"Loaded SelFT grad dictionary from '{grad_dict_path}'")
    else:
        # Get gradient for trainable parameters
        grad_dict = selft_get_score(unet, target_concepts, anchor_concepts, loss, device, text_encoder, tokenizer, pipe=diffusion_pipe)
        
        # Save the gradient dictionary
        if grad_dict_path is not None:
            os.makedirs(os.path.dirname(grad_dict_path), exist_ok=True)
            torch.save(grad_dict, grad_dict_path)
            print(f"Saved SelFT grad_dict to '{grad_dict_path}'")
    
    mask_dict = build_global_topk_mask_dict(unet, grad_dict, topk)
    
    # Save the importance mask dictionary 
    if mask_dict_path is not None:
        os.makedirs(os.path.dirname(mask_dict_path), exist_ok=True)
        torch.save(mask_dict, mask_dict_path)
        print(f"Saved SelFT mask_dict to '{mask_dict_path}'")
        
    return mask_dict


def apply_selft_masks(unet, selft_mask_dict):
    """Apply SelFT masks by registering hooks on parameter gradients"""
    # Remove any existing hooks
    grad_hooks = []
    
    # Function to create the masking hook
    def make_hook(mask, param_name):
        def hook(grad):
            
            # Apply binary bask to gradient
            masked_grad = grad * mask
            
            # Testing to make sure hooks are activating
            # print(f"   Hook activated for parameter: {param_name}")
            # print(f"   Gradient shape: {grad.shape}, Mask shape: {mask.shape}")
            # print(f"   Gradient L2 norm before masking: {torch.norm(grad).item()}")
            # print(f"   Gradient L2 norm after masking: {torch.norm(masked_grad).item()}")
            
            return masked_grad
        return hook
    
    # Register new hooks
    hook_count = 0
    for name, param in unet.named_parameters():
        if name in selft_mask_dict and param.requires_grad:
            # Create a specific hook for this parameter with its corresponding mask
            mask_tensor = selft_mask_dict[name].to(param.device)
            hook = param.register_hook(make_hook(mask_tensor, name))
            grad_hooks.append(hook)
            hook_count += 1
    
    print(f"Registered {hook_count} gradient masking hooks")
    return grad_hooks