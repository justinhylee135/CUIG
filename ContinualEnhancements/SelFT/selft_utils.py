# Standard Library
import os
import ast

# Third Party
import torch
from tqdm import tqdm
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def apply_selft_masks(unet, selft_mask_dict):
    """Apply SelFT masks by registering hooks on parameter gradients"""
    # Remove any existing hooks
    grad_hooks = []
    
    # Function to create the masking hook
    def make_hook(mask, param_name):
        def hook(grad):
            # Apply mask - zero out gradients we want to ablate
            # print(f"⚡ Hook activated for parameter: {param_name}")
            # print(f"   Gradient shape: {grad.shape}, Mask shape: {mask.shape}")
            # print(f"   Gradient L2 norm before masking: {torch.norm(grad).item()}")
            
            # This is where you could set a breakpoint in a debugger
            masked_grad = grad * mask
            
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

def encode_text_diffusers(text_encoder, tokenizer, prompt, device):
    """Encode text using diffusers text encoder"""
    if isinstance(prompt, str):
        prompt = [prompt]
        
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]
    
    return text_embeddings

def get_selft_mask_dict(pipeline, mask_dict_path, grad_dict_path, prompt_list, anchor, topk, loss, device):
    """Get the SelFT mask for the UNet model"""  
    # Check if the mask dictionary exists
    if mask_dict_path is not None and os.path.exists(mask_dict_path):
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
        grad_dict = selft_get_score(pipeline.unet, prompt_list, anchor, loss, device, pipeline.text_encoder, pipeline.tokenizer)
        
        # Save the gradient dictionary
        if grad_dict_path is not None:
            os.makedirs(os.path.dirname(grad_dict_path), exist_ok=True)
            torch.save(grad_dict, grad_dict_path)
            print(f"Saved SelFT grad_dict to '{grad_dict_path}'")
    
    # Flattened list of all scores
    mask_dict = {}
    global_scores = []
    param_shapes = {}
    param_slices = {}

    total_params = 0
    selected_params = 0
    total_elements = 0
    selected_elements = 0

    # Collect all |param*grad| values into a single vector
    for name, param in pipeline.unet.named_parameters():
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
    for name, (start, end) in param_slices.items():
        flat_mask = global_mask[start:end]
        if flat_mask.any():  # if at least one element selected
            selected_params += 1

    # Output statistics
    print(f"Selected {selected_params} of {total_params} parameters or {(selected_params / total_params):.2%}")
    print(f"Selected {selected_elements:,} of {total_elements:,} elements or {(selected_elements / total_elements):.2%}")

    # Create final mask_dict
    for name, param in pipeline.unet.named_parameters():
        if not param.requires_grad:
            mask = torch.zeros_like(param)
        elif name in param_shapes:
            shape = param_shapes[name]
            start, end = param_slices[name]
            flat_mask = global_mask[start:end]
            mask = flat_mask.view(shape)
        else:
            # Not among selected group → full zero mask
            mask = torch.zeros_like(param)

        mask_dict[name] = mask.bool()
    
    # Save the importance mask dictionary 
    if mask_dict_path is not None:
        os.makedirs(os.path.dirname(mask_dict_path), exist_ok=True)
        torch.save(mask_dict, mask_dict_path)
        print(f"Saved SelFT mask_dict to {mask_dict_path}")
        
    return mask_dict

def selft_get_score(unet, prompt_list, anchor, loss, device, text_encoder, tokenizer):  
    """Compute SelFT scores using diffusers components"""
    # Create scheduler
    ddim_steps = 50
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(ddim_steps, device=device)

    # Create list of gradient dictionary 
    all_grad_dicts = []
    
    # Inference parameters
    start_guidance = 3
    negative_guidance = 1.0
    
    # Anchor list can be a single anchor or a list of anchors
    anchor_list = process_anchor(anchor, prompt_list)
    
    # Iterate through each prompt and average gradients
    num_prompts = len(prompt_list)
    for prompt_idx in range(num_prompts):
        grad_dict = {}
        for name, param in unet.named_parameters():  
            if param.requires_grad:                                                                                                                 
                grad_dict[name] = 0
            
        # Pull target-anchor pair
        target = prompt_list[prompt_idx]
        anchor = anchor_list[prompt_idx]
        
        with torch.no_grad():            
            # Wrap in list
            anchor = [anchor]
            target = [target]
            
            # Get text embeddings
            print(f"Iteration {prompt_idx + 1}/{num_prompts}")
            print(f"\nSelFT target: '{target}'")
            print(f"SelFT anchor: '{anchor}'")
            c_target = encode_text_diffusers(text_encoder, tokenizer, target, device)
            c_anchor = encode_text_diffusers(text_encoder, tokenizer, anchor, device)
            uc = encode_text_diffusers(text_encoder, tokenizer, [""], device)

        # Prepare shape for sampling
        shape = [1, 4, 512 // 8, 512 // 8]

        # Get timesteps
        timesteps = scheduler.timesteps

        # Iterate through timesteps
        progress_bar = tqdm(range(min(ddim_steps, len(timesteps))), desc="Training")
        for t_idx in progress_bar:
            with torch.no_grad():
                # Initialize random latents for first iteration
                if t_idx == 0:
                    latents = torch.randn(shape, device=device, dtype=unet.dtype)
                    latents = latents * scheduler.init_noise_sigma
                    current_timestep = timesteps[0]
                else:
                    # Start from random noise and partially denoise
                    latents = torch.randn(shape, device=device, dtype=unet.dtype)
                    latents = latents * scheduler.init_noise_sigma
                    
                    # Partially denoise until the current timestep
                    for step_idx in range(t_idx):
                        timestep_tensor = timesteps[step_idx].unsqueeze(0).to(device)
                        
                        # Expand latents for classifier free guidance
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps[step_idx])
                        
                        # Concatenate embeddings for CFG
                        encoder_hidden_states = torch.cat([uc, c_target])
                        
                        # Predict noise
                        noise_pred = unet(latent_model_input, timestep_tensor, encoder_hidden_states=encoder_hidden_states).sample
                        
                        # Perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + start_guidance * (noise_pred_text - noise_pred_uncond)
                        
                        # Compute previous sample
                        latents = scheduler.step(noise_pred, timesteps[step_idx], latents).prev_sample
                        
                    current_timestep = timesteps[t_idx]
            
            # Convert to tensor
            timestep_tensor = current_timestep.unsqueeze(0).to(device)
            
            # Enable gradient tracking
            latents = latents.detach().requires_grad_(True)
            
            # Forward pass with both target and anchor prompts
            unet.zero_grad()
        
            assert latents.shape[0] == c_target.shape[0], "Batch size mismatch between latents and conditioning"

            with torch.no_grad():
                # Get anchor and uc noise prediction (without gradient)
                noise_unconditional = unet(latents, timestep_tensor, encoder_hidden_states=uc).sample
                noise_anchor = unet(latents, timestep_tensor, encoder_hidden_states=c_anchor).sample

            with torch.enable_grad():
                # Predicted noise for the target prompt
                noise_target = unet(latents, timestep_tensor, encoder_hidden_states=c_target).sample
            
                # Compute loss
                if loss == 'ca':
                    obj = ((noise_target - noise_anchor.detach()) ** 2).sum()
                elif loss == 'esd':
                    criteria = torch.nn.MSELoss()
                    reverse_cfg_noise = noise_unconditional - (negative_guidance * (noise_anchor - noise_unconditional))
                    obj = criteria(noise_target, reverse_cfg_noise.detach())
                else:
                    raise ValueError(f"Unsupported SelFT loss type: {loss}")
            
            # Backpropagate
            obj.backward()
            progress_bar.set_postfix(loss=obj.item())
            
            # Accumulate gradients
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    grad_dict[name] += param.grad.detach().clone()
            
            # Clean up gradients for next iteration
            unet.zero_grad()

        # Calculate and print the L2 norm for each item in grad_dict
        print(f"'{len(grad_dict)}' Gradient L2 norms for '{target}' to '{anchor}':")
        for name, grad in grad_dict.items():
            l2_norm = torch.norm(grad).item()
            print(f"\t{name}: {l2_norm}")
        
        # Append the gradient dictionary for this target-anchor pair
        all_grad_dicts.append(grad_dict)
    
    # Average gradients across all prompts
    print(f"\nAveraging gradients across '{num_prompts}' prompts...")
    averaged_grad_dict = {}
    
    # Initialize with zeros
    for name, param in unet.named_parameters():
        if param.requires_grad:
            averaged_grad_dict[name] = torch.zeros_like(param)
    
    # Sum gradients from all prompts
    for grad_dict in all_grad_dicts:
        for name in averaged_grad_dict.keys():
            averaged_grad_dict[name] += grad_dict[name]
    
    # Divide by number of prompts to get average
    for name in averaged_grad_dict.keys():
        averaged_grad_dict[name] = averaged_grad_dict[name] / num_prompts

    # Print final averaged L2 norms
    print(f"\n'{len(averaged_grad_dict)}' Final averaged L2 norms:")
    for name, grad in averaged_grad_dict.items():
        l2_norm = torch.norm(grad).item()
        print(f"\t{name}: {l2_norm:.6f}")

    return averaged_grad_dict

def process_anchor(anchor, prompt_list):
    """Process anchor to ensure it matches the prompt list length"""
    # Anchor list can be a single anchor or a list of anchors
    if "[" in anchor:
        anchor_list = ast.literal_eval(anchor)
    else:
        anchor_list = [anchor]

    # Broadcast anchor list to match prompt list length
    if len(anchor_list) <= len(prompt_list):
        anchor_list = anchor_list * len(prompt_list)
    
    # If anchor still doesn't match prompt list length, raise an error
    if len(anchor_list) != len(prompt_list):
        raise ValueError(f"Anchor list length {len(anchor_list)} does not match prompt list length {len(prompt_list)}")

    return anchor_list