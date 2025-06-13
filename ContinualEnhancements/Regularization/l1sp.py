import torch

def calculate_l1sp_loss(model, original_params):
    # Store original parameters for first iteration
    if not original_params:
        print(f"Storing original parameters for L1-SP regularization")
        for name, module in model.named_modules():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    original_params[name] = param.detach().clone().requires_grad_(False)
            
        # Count total number of parameters elements for normalization
        total_elements = sum(p.numel() for p in original_params.values())
        print(f"Stored {len(original_params)} original parameters ({total_elements:,} total values) for L1-SP regularization")
        
    l1sp_loss = 0.0
    elements_included = 0
    # Calculcate L1-SP loss
    for name, module in model.named_modules():
        for name, param in module.named_parameters():
            if name in original_params:
                # Ensure same dtype and device
                orig_param = original_params[name].to(dtype=param.dtype).to(param.device)

                # Calculate absolute difference
                abs_diff = torch.sum(torch.abs(param - orig_param))

                # Accumulate loss and parameter element count
                elements_included += param.numel()
                l1sp_loss += abs_diff
    
    # Normalize loss by number of elements included
    if elements_included > 0:
        l1sp_loss /= elements_included

    return l1sp_loss