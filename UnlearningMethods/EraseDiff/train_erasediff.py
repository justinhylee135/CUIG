# ref
# - https://github.com/JingWu321/EraseDiff/tree/master/sd
import os
import sys
import ast
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything

sys.path.append('.')
import argparse
import gc
from timm.utils import AverageMeter

# CHANGE 1: Import diffusers components
from diffusers import (
    StableDiffusionPipeline
)

# Import dataset utilities
from dataset import setup_forget_style_data
from constants.const import theme_available, class_available

# ADDITION: Import Continual Enhancement utilities
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
## Utils
from utils import (
    get_param,
    set_param,
    setup_diffusers_model,
    encode_prompts,
    encode_images, 
    get_trainable_params,
    create_anchor_prompts
)
## Simultaneous
from ContinualEnhancements.Simultaneous.sim_utils import sample_and_evaluate_ua, check_early_stopping
## Regularization
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss
from ContinualEnhancements.Regularization.inverse_ewc import (
    accumulate_fisher,
    calculate_inverse_ewc_loss,
    load_fisher_information,
    save_fisher_information
)
from ContinualEnhancements.Regularization.trajectory import (
    calculate_trajectory_loss,
    load_delta_from_path,
    save_delta_to_path
)
## SelFT
from ContinualEnhancements.SelFT.selft_utils import (
    get_selft_mask_dict,
    apply_selft_masks
)
## Projection
from ContinualEnhancements.Projection.gradient_projection import (
    generate_gradient_projection_prompts,
    get_anchor_embeddings,
    apply_gradient_projection
)


def erasediff(
    concept,
    anchor,
    concept_type,
    data_method,
    forget_data_dir,
    remain_data_dir,
    output_path,
    base_model_dir,
    unet_ckpt,
    K_steps,
    train_method,
    lambda_bome=0.1,
    remain_weight=0.0,
    batch_size=4,
    cycles=5,
    lr=1e-5,
    device="cuda:0",
    image_size=512,
    seed=42,
    verbose=False,
    # Continual Enhancement parameters
    eval_every=None,
    eval_start=0,
    patience=2000,
    stop_threshold=99.0,
    classifier_dir=None,
    l1sp_weight=0.0,
    l2sp_weight=0.0,
    inverse_ewc_lambda=0.0,
    inverse_ewc_use_l2=False,
    previous_fisher_path=None,
    save_fisher_path=None,
    trajectory_lambda=0.0,
    previous_delta_path=None,
    save_delta_path=None,
    set_original_params_to_base=False,
    selft_loss=None,
    selft_topk=0.01,
    selft_anchor="",
    selft_grad_dict_path=None,
    selft_mask_dict_path=None,
    with_gradient_projection=False,
    gradient_projection_prompts=None,
    gradient_projection_num_prompts=400,
    previously_unlearned=None,
):
    """
    EraseDiff with Continual Enhancements
    """
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Setup diffusers model
    unet, vae, text_encoder, tokenizer, scheduler = setup_diffusers_model(base_model_dir, device)
    
    # Load base model for original parameters if needed
    original_params = {}
    if set_original_params_to_base:
        print(f"Storing original unet from '{base_model_dir}' for regularization")
        for name, param in unet.named_parameters():
            original_params[name] = param.detach().clone().requires_grad_(False)
        print(f"\tStored '{len(original_params)}' original parameters from base model")
    
    # Load unet ckpt if given
    if unet_ckpt is not None:
        if not os.path.exists(unet_ckpt):
            print(f"UNet checkpoint not found at '{unet_ckpt}'. Using default UNet from pipeline '{base_model_dir}'...")
        else:
            print(f"Loading UNet checkpoint from '{unet_ckpt}'")
            unet_sd = torch.load(unet_ckpt, map_location='cpu')
            missing, unexpected = unet.load_state_dict(unet_sd, strict=False)

            print(f"UNet checkpoint '{unet_ckpt}' loaded to StableDiffusion pipeline")
            print(f"\tLoaded '{len(unet_sd)}' keys with '{len(missing)}' missing and '{len(unexpected)}' unexpected keys.")

    
    # Get trainable parameters
    train_param_names, parameters = get_trainable_params(unet, train_method)
    print(f"\tTraining {len(parameters)} parameters with method: '{train_method}'")
    for name, param in unet.named_parameters():
        if name in train_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Setup data loaders
    forget_dl, remain_dl = setup_forget_style_data(data_method, forget_data_dir, remain_data_dir, batch_size, image_size)
    print(f"Data Loaders...")
    print(f"\tUsing batch size: '{batch_size}'")
    print(f"\tForget dataset # batches: {len(forget_dl)}")
    print(f"\tRemain dataset # batches: {len(remain_dl)}")

    # Set up optimizer
    print(f"Training Settings...")
    print(f"\tUsing Learning rate: '{lr}' and lambda_bome: '{lambda_bome}' and remain_weight: '{remain_weight}'")
    print(f"\tTraining for '{cycles}' cycles with '{K_steps}' K-steps and '{args.steps_per_cycle}' steps per cycle")
    print(f"Setting up Adam optimizer with lr '{lr}'")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    
    # Setup for early stopping
    if eval_every is not None:
        if classifier_dir is None:
            raise ValueError("Classifier directory must be specified for early stopping evaluation.")
        print(f"Using early stopping with patience '{patience}' and eval_every '{eval_every}'")
        best_ua = 0.0
        no_improvement_count = 0
    
    # Setup regularization
    if l1sp_weight > 0.0:
        print(f"Using l1sp loss with '{l1sp_weight}'")
    if l2sp_weight > 0.0:
        print(f"Using l2sp loss with '{l2sp_weight}'")

    # Inverse EWC
    current_fisher = {}
    previous_aggregated_fisher = None
    if inverse_ewc_lambda > 0.0:
        print(f"Using Inverse EWC regularizer with lambda '{inverse_ewc_lambda}'")
        if inverse_ewc_use_l2:
            print(f"\tUsing L2 distance for Inverse EWC loss")
        else:
            print(f"\tUsing L1 distance for Inverse EWC loss")
        
        if previous_fisher_path is not None and os.path.exists(previous_fisher_path):
            print(f"\tLoading previous fisher information from '{previous_fisher_path}'")
            previous_aggregated_fisher = load_fisher_information(previous_fisher_path, device)
        else:
            inverse_ewc_lambda = 0.0
            print(f"\tNo previous fisher information found. Turning off Inverse EWC regularization...")
    
    # Setup trajectory regularization
    previous_aggregated_delta = None
    if trajectory_lambda > 0.0:
        print(f"Using Trajectory regularizer with lambda '{trajectory_lambda}'")
        if previous_delta_path is not None and os.path.exists(previous_delta_path):
            print(f"\tLoading previous parameter deltas from '{previous_delta_path}'")
            previous_aggregated_delta = load_delta_from_path(previous_delta_path, device)
        else:
            print(f"\tNo previous parameter deltas found. Turning off Trajectory regularization...")
            trajectory_lambda = 0.0
    
    # Setup SelFT
    selft_mask_dict = None
    grad_hooks = []
    if selft_loss is not None:
        print(f"Using SelFT with loss type: '{selft_loss}', top-k: '{selft_topk}'")
        
        prompt_list = []
        anchor_list = []
        for i in range(len(concept)):
            # Get prompts for SelFT (using concept or concept)
            if concept_type == "style":
                prompt_list.append(f"{concept[i].replace('_', ' ')} style")
                anchor_list.append(f"{anchor[i].replace('Seed_Images', 'Photo').replace('_', ' ')} style")
            elif concept_type == "object":
                prompt_list.append(f"An image of {concept[i]}")
                anchor_list.append(f"An image of {anchor[i]}")
            else:
                raise ValueError(f"Invalid concept_type '{concept_type}' for SelFT. Must be 'style' or 'object'.")
        
        unet.eval()
        selft_mask_dict = get_selft_mask_dict(
            unet, text_encoder, tokenizer,
            selft_mask_dict_path, selft_grad_dict_path,
            prompt_list, selft_anchor, selft_topk,
            selft_loss, device
        )
        
        grad_hooks = apply_selft_masks(unet, selft_mask_dict)
        print(f"Applied SelFT masks with '{len(grad_hooks)}' hooks")
    
    # Setup gradient projection
    anchor_embeddings_matrix = None
    if with_gradient_projection:
        print(f"\nUsing gradient projection to preserve anchor concepts.")
        proj_anchor_prompts = []
        
        if gradient_projection_prompts:
            if os.path.isfile(gradient_projection_prompts):
                with open(gradient_projection_prompts, 'r') as f:
                    prompts = [line.strip() for line in f.readlines() if line.strip()]
                print(f"\tLoaded '{len(prompts)}' anchor prompts from '{gradient_projection_prompts}'")
                proj_anchor_prompts.extend(prompts)
            else:
                print(f"Generating gradient projection prompts")
                prompt_list = []
                for c in concept:
                    # Get prompts for SelFT (using concept or concept)
                    if concept_type == "style":
                        prompt_list.append(f"{c.replace('_', ' ')} Style")
                    elif concept_type == "object":
                        prompt_list.append(f"An image of {c}")
                    else:
                        raise ValueError(f"Invalid concept_type '{concept_type}' for SelFT. Must be 'style' or 'object'.")
                        
                proj_anchor_prompts = generate_gradient_projection_prompts(
                    file_path=gradient_projection_prompts,
                    num_prompts=gradient_projection_num_prompts,
                    concept_type=concept_type,
                    previously_unlearned=previously_unlearned,
                    target_concept_list=prompt_list
                )
        
        if proj_anchor_prompts:
            print(f"Total anchor prompts collected: '{len(proj_anchor_prompts)}'")
            anchor_embeddings_matrix = get_anchor_embeddings(
                proj_anchor_prompts, text_encoder, tokenizer, device
            )
    
    # TRAINING LOOP
    total_steps = 0
    stop_training = False
    
    for cycle in range(cycles):
        if stop_training:
            break
            
        unet.train()
        
        with tqdm(total=K_steps*args.steps_per_cycle + args.steps_per_cycle, desc=f'Cycle {cycle}') as pbar:
            # Store parameters before Phase 1
            param_i = get_param(unet)  # get \theta_i
            
            # Phase 1: K-steps of forgetting
            for j in range(K_steps):
                unl_losses = AverageMeter()

                for i in range(args.steps_per_cycle):
                    optimizer.zero_grad()
                    
                    forget_images, forget_prompts = next(iter(forget_dl))
                    if data_method == 'erasediff':
                        anchor_prompts = create_anchor_prompts(forget_prompts, concept, anchor, concept_type)
                    elif data_method == 'ca':
                        if isinstance(forget_prompts, tuple): forget_prompts = list(forget_prompts)
                        anchor_prompts = forget_prompts.copy()
                        for i, prompt in enumerate(forget_prompts):
                            if concept_type == 'style':
                                concept_idx = np.random.choice(len(concept))
                                c_unlearn = concept[concept_idx].replace('_', ' ')
                                augment_idx = np.random.choice([0,1])
                                forget_prompts[i] = (f"{prompt}, in {c_unlearn} Style" if augment_idx == 0 else f"In {c_unlearn} Style, {prompt}")
                            elif concept_type == 'object':
                                idx_options = []
                                for j, anc in enumerate(anchor):
                                    if anc in prompt:
                                        idx_options.append(j)
                                concept_idx = np.random.choice(idx_options)
                                selected_target = concept[concept_idx]
                                selected_anchor = anchor[concept_idx]
                                replacement_prompt = prompt.replace(selected_anchor, selected_target)
                                if forget_prompts[i] == replacement_prompt:
                                    print(f"Replacement unsuccessful for prompt '{prompt}' and anchor '{selected_anchor}'")
                                    replacement_prompt = f"An image of {selected_target}"
                                forget_prompts[i] = replacement_prompt
                    
                    if isinstance(forget_prompts, tuple):
                        forget_prompts = list(forget_prompts)
                    if isinstance(anchor_prompts, tuple):
                        anchor_prompts = list(anchor_prompts)
                    
                    if verbose: print(f"{i}. \n\tForget prompts: '{forget_prompts}'\n\tAnchor prompts: '{anchor_prompts}'")
                
                    # Process data with diffusers
                    forget_images = forget_images.to(device)
                    forget_emb = encode_prompts(forget_prompts, tokenizer, text_encoder, device)
                    anchor_emb = encode_prompts(anchor_prompts, tokenizer, text_encoder, device)
                    forget_latents = encode_images(forget_images, vae)
                    
                    # Sample noise and timesteps
                    noise = torch.randn_like(forget_latents, device=device)
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps,
                        (forget_latents.shape[0],), device=device
                    ).long()
                    
                    # Add noise and predict
                    forget_noisy = scheduler.add_noise(forget_latents, noise, timesteps)
                    forget_pred = unet(forget_noisy, timesteps, forget_emb).sample
                    anchor_pred = unet(forget_noisy, timesteps, anchor_emb).sample.detach() # Stop Gradient Applied
                    
                    forget_loss = criteria(forget_pred, anchor_pred)
                    forget_loss.backward()
                    
                    optimizer.step()
                    unl_losses.update(forget_loss)
                    
                    loss_dict = {'avg_k_loss': unl_losses.avg.item()}
                    loss_dict['forget[0]'] = forget_prompts[0]
                    pbar.set_postfix(loss_dict)
                    pbar.update(1)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Reset parameters to before Phase 1
            unet = set_param(unet, param_i)  
            avg_k_loss = unl_losses.avg.detach()

            # Phase 2: Update with remain loss and forget loss
            for i in range(args.steps_per_cycle):
                unet.train()
                optimizer.zero_grad()
                
                forget_images, forget_prompts = next(iter(forget_dl))
                if remain_weight > 0.0: 
                    remain_images, remain_prompts = next(iter(remain_dl))
                else:
                    remain_prompts = []

                if data_method == 'erasediff':
                    anchor_prompts = create_anchor_prompts(forget_prompts, concept, anchor, concept_type)
                elif data_method == 'ca':
                    if isinstance(forget_prompts, tuple): forget_prompts = list(forget_prompts)
                    anchor_prompts = forget_prompts.copy()
                    for i, prompt in enumerate(forget_prompts):
                        if concept_type == 'style':
                            concept_idx = np.random.choice(len(concept))
                            c_unlearn = concept[concept_idx].replace('_', ' ')
                            augment_idx = np.random.choice([0,1])
                            forget_prompts[i] = (f"{prompt}, in {c_unlearn} Style" if augment_idx == 0 else f"In {c_unlearn} Style, {prompt}")
                        elif concept_type == 'object':
                            idx_options = []
                            for j, anc in enumerate(anchor):
                                if anc in prompt:
                                    idx_options.append(j)
                            concept_idx = np.random.choice(idx_options)
                            selected_target = concept[concept_idx]
                            selected_anchor = anchor[concept_idx]
                            replacement_prompt = prompt.replace(selected_anchor, selected_target)
                            if forget_prompts[i] == replacement_prompt:
                                print(f"Replacement unsuccessful for prompt '{prompt}' and anchor '{selected_anchor}'")
                                replacement_prompt = f"An image of {selected_target}"
                            forget_prompts[i] = replacement_prompt

                if isinstance(forget_prompts, tuple):
                    forget_prompts = list(forget_prompts)
                if remain_weight > 0.0 and isinstance(remain_prompts, tuple):
                    remain_prompts = list(remain_prompts)
                if isinstance(anchor_prompts, tuple):
                    anchor_prompts = list(anchor_prompts)
                
                if verbose: print(f"{i}. \n\tForget prompts: '{forget_prompts}'\n\tAnchor prompts: '{anchor_prompts}'\n\tRemain prompts: '{remain_prompts}'")
                
                # Remain stage - compute standard diffusion loss
                if remain_weight > 0.0:
                    remain_images = remain_images.to(device)
                    remain_latents = encode_images(remain_images, vae)
                    remain_emb = encode_prompts(remain_prompts, tokenizer, text_encoder, device)
                    
                    noise = torch.randn_like(remain_latents, device=device)
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps,
                        (remain_latents.shape[0],), device=device
                    ).long()
                    
                    noisy_latents = scheduler.add_noise(remain_latents, noise, timesteps)
                    noise_pred = unet(noisy_latents, timesteps, remain_emb).sample
                    remain_loss = criteria(noise_pred, noise)
                
                # Forget stage
                forget_images = forget_images.to(device)
                forget_latents = encode_images(forget_images, vae)
                forget_emb = encode_prompts(forget_prompts, tokenizer, text_encoder, device)
                anchor_emb = encode_prompts(anchor_prompts, tokenizer, text_encoder, device)
                
                noise = torch.randn_like(forget_latents, device=device)
                forget_timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (forget_latents.shape[0],), device=device
                ).long()
                
                forget_noisy = scheduler.add_noise(forget_latents, noise, forget_timesteps)
                forget_out = unet(forget_noisy, forget_timesteps, forget_emb).sample
                anchor_out = unet(forget_noisy, forget_timesteps, anchor_emb).sample.detach()
                
                # Target to Anchor loss
                unlearn_loss = criteria(forget_out, anchor_out)
                
                # Adjusted unlearning loss should be at least as great as avg_k_loss
                q_loss = unlearn_loss - avg_k_loss
                
                # Compute total loss with regularizers
                total_loss = lambda_bome * q_loss
                if remain_weight > 0.0:
                    total_loss += remain_weight * remain_loss
                erasediff_loss = total_loss.detach().clone()

                # Add regularization losses
                if l1sp_weight > 0.0:
                    l1sp_loss = l1sp_weight * calculate_l1sp_loss(unet, original_params)
                    total_loss += l1sp_loss
                
                if l2sp_weight > 0.0:
                    l2sp_loss = l2sp_weight * calculate_l2sp_loss(unet, original_params)
                    total_loss += l2sp_loss
                
                if inverse_ewc_lambda > 0.0 and previous_aggregated_fisher is not None:
                    inverse_ewc_loss = inverse_ewc_lambda * calculate_inverse_ewc_loss(
                        unet, previous_aggregated_fisher, original_params,
                        device, use_l2=inverse_ewc_use_l2
                    )
                    total_loss += inverse_ewc_loss
                
                if trajectory_lambda > 0.0 and previous_aggregated_delta is not None:
                    trajectory_loss = trajectory_lambda * calculate_trajectory_loss(
                        unet, previous_aggregated_delta, original_params, device
                    )
                    total_loss += trajectory_loss
                
                # Backprop
                total_loss.backward()
                
                # Accumulate Fisher
                if save_fisher_path is not None:
                    current_fisher = accumulate_fisher(unet, current_fisher)
                
                # Apply gradient projection
                if with_gradient_projection and anchor_embeddings_matrix is not None:
                    apply_gradient_projection(unet, anchor_embeddings_matrix, device)
                
                # Optimizer step
                optimizer.step()
                
                # Prepare loss dict for progress bar
                loss_dict = {'total': total_loss.item()}
                loss_dict['q_loss'] = q_loss.item()
                loss_dict['forget[0]'] = forget_prompts[0]
                if remain_weight > 0.0:
                    loss_dict['remain'] = remain_loss.item()
                if l1sp_weight > 0.0:
                    loss_dict['l1sp'] = l1sp_loss.item()
                if l2sp_weight > 0.0:
                    loss_dict['l2sp'] = l2sp_loss.item()
                if inverse_ewc_lambda > 0.0 and previous_aggregated_fisher is not None:
                    loss_dict['inv_ewc'] = inverse_ewc_loss.item()
                if trajectory_lambda > 0.0 and previous_aggregated_delta is not None:
                    loss_dict['traj'] = trajectory_loss.item()
                if len(loss_dict) > 2:
                    loss_dict['erasediff_loss'] = erasediff_loss.item()

                pbar.set_postfix(loss_dict)
                pbar.update(1)
                
                # Early stopping evaluation
                total_steps += 1
                if eval_every is not None and total_steps % eval_every == 0 and total_steps >= eval_start:
                    unet.eval()

                    # Create pipeline for evaluation
                    eval_pipeline = StableDiffusionPipeline(
                        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                        unet=unet, scheduler=scheduler, safety_checker=None,
                        feature_extractor=None, requires_safety_checker=False
                    )
                    
                    ua = sample_and_evaluate_ua(
                        eval_pipeline, concept_type, total_steps, output_path,
                        concept, forget_prompts[0], device, classifier_dir
                    )
                    unet.train()
                    print(f"Step '{total_steps}', Unlearned Accuracy: '{ua}'")
                    
                    # Check for early stopping
                    best_ua, no_improvement_count, stop_training = check_early_stopping(
                        ua, best_ua, no_improvement_count, eval_every, patience, stop_threshold
                    )
                    
                    if stop_training:
                        print(f"Early stopping triggered at step {total_steps}")
                        break
    
    # Clean up gradient hooks
    if grad_hooks:
        for hook in grad_hooks:
            hook.remove()
        print(f"Removed '{len(grad_hooks)}' gradient hooks")
    
    # Save Fisher Information
    if save_fisher_path is not None and current_fisher:
        save_fisher_information(current_fisher, save_fisher_path, total_steps, previous_aggregated_fisher)
    
    # Save parameter delta
    if save_delta_path is not None:
        save_delta_to_path(unet, original_params, save_delta_path, previous_aggregated_delta)
    
    unet.eval()
    return unet, vae, text_encoder, tokenizer, scheduler, train_param_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='EraseDiffEnhanced',
        description='EraseDiff with Continual Enhancements using Diffusers')
    
    # Basic EraseDiff parameters
    parser.add_argument('--forget_data_dir', help='forget data dir', type=str, required=False, default='data')
    parser.add_argument('--remain_data_dir', help='remain data dir', type=str, required=False, default='data')
    parser.add_argument('--data_method', help='method to get data', type=str, default='erasediff')
    parser.add_argument('--concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--anchor', help='anchor for remain data', type=str, required=True)
    parser.add_argument('--concept_type', type=str, default='style', 
                       choices=['style', 'object', 'celebrity'])
    parser.add_argument('--train_method', help='method of training', type=str, default="xattn",
                       choices=["noxattn", "selfattn", "xattn", "kv-xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--lambda_bome', help='weight for BOME algorithm', type=float, default=0.1)
    parser.add_argument('--remain_weight', help='weight for remain loss', type=float, default=0.0)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4)
    parser.add_argument('--cycles', help='cycles to train', type=int, default=5)
    parser.add_argument('--steps_per_cycle', help='steps per cycle', type=int, default=100)
    parser.add_argument('--K_steps', type=int, default=2)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument('--base_model_dir', help='HuggingFace model ID or path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--unet_ckpt', help='Base model for original params', type=str, default=None)
    parser.add_argument('--output_path', help='output path for model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--overwrite_existing_ckpt', help='Overwrite existing checkpoint if it exists', action='store_true', default=False)
        
    # Continual Enhancement parameters
    ## Simultaneous (Early Stopping)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--classifier_dir', type=str, default=None)
    parser.add_argument('--stop_threshold', type=float, default=99.0)
    
    ## Regularizers
    parser.add_argument('--l1sp_weight', type=float, default=0.0)
    parser.add_argument('--l2sp_weight', type=float, default=0.0)
    parser.add_argument('--set_original_params_to_base', action='store_true')
    
    ### Inverse EWC
    parser.add_argument('--inverse_ewc_lambda', type=float, default=0.0)
    parser.add_argument('--inverse_ewc_use_l2', action='store_true')
    parser.add_argument('--previous_fisher_path', type=str, default=None)
    parser.add_argument('--save_fisher_path', type=str, default=None)
    
    ### Trajectory
    parser.add_argument('--trajectory_lambda', type=float, default=0.0)
    parser.add_argument('--previous_delta_path', type=str, default=None)
    parser.add_argument('--save_delta_path', type=str, default=None)
    
    ## SelFT
    parser.add_argument('--selft_loss', type=str, default=None, choices=['esd', 'ca'])
    parser.add_argument('--selft_topk', type=float, default=0.01)
    parser.add_argument('--selft_anchor', type=str, default="")
    parser.add_argument('--selft_grad_dict_path', type=str, default=None)
    parser.add_argument('--selft_mask_dict_path', type=str, default=None)
    
    ## Projection
    parser.add_argument('--with_gradient_projection', action='store_true')
    parser.add_argument('--gradient_projection_prompts', type=str, default=None)
    parser.add_argument('--gradient_projection_num_prompts', type=int, default=400)
    parser.add_argument('--previously_unlearned', type=str, default=None)
    
    args = parser.parse_args()
    
    # Output directory
    output_parent_dir = Path(args.output_path).parent
    os.makedirs(output_parent_dir, exist_ok=True)
    print(f"Creating output parent directory at '{output_parent_dir}'")
    
    # Exit if model exists
    if os.path.exists(args.output_path):
        if args.overwrite_existing_ckpt:
            print(f"Overwriting existing model at {args.output_path}.")
        else:
            print(f"Model already exists at '{args.output_path}'. Set flag '--overwrite_existing_ckpt' to overwrite it.")
            sys.exit(0)
    
    # Set up target and anchor lists
    print(f"args.concept: '{args.concept}', args.anchor: '{args.anchor}'")
    if "[" in args.concept:
        concept = ast.literal_eval(args.concept)
    else: 
        concept = [args.concept]
    if "[" in args.anchor:
        anchor = ast.literal_eval(args.anchor)
    else:
        anchor = [args.anchor]
    if len(anchor) < len(concept) and (len(concept) % len(anchor) == 0):
        print(f"Broadcasting anchor list '{anchor}' by a factor of '{len(concept) // len(anchor)}'")
        anchor = anchor * (len(concept) // len(anchor))
    for i in range(len(concept)):
        print(f"{i+1}. Mapping concept '{concept[i]}' to anchor '{anchor[i]}'")

    # Setup data directories
    forget_data_dir = []
    remain_data_dir = []
    if args.data_method == 'erasediff':
        for i in range(len(concept)):
            forget_data_dir.append(os.path.join(args.forget_data_dir, concept[i]))
            remain_data_dir.append(os.path.join(args.remain_data_dir, anchor[i]))
    elif args.data_method == 'ca':
        if "[" in args.forget_data_dir:
            forget_data_dir = ast.literal_eval(args.forget_data_dir)
        else: 
            forget_data_dir = [args.forget_data_dir]
        if "[" in args.remain_data_dir:
            remain_data_dir = ast.literal_eval(args.remain_data_dir)
        else:
            remain_data_dir = [args.remain_data_dir]


    # Run enhanced EraseDiff
    unet, vae, text_encoder, tokenizer, scheduler, train_param_names = erasediff(
        concept=concept,
        anchor=anchor,
        concept_type=args.concept_type,
        data_method=args.data_method,
        forget_data_dir=forget_data_dir,
        remain_data_dir=remain_data_dir,
        output_path=args.output_path,
        base_model_dir=args.base_model_dir,
        unet_ckpt=args.unet_ckpt,
        K_steps=args.K_steps,
        train_method=args.train_method,
        lambda_bome=args.lambda_bome,
        remain_weight=args.remain_weight,
        batch_size=args.batch_size,
        cycles=args.cycles,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
        # Continual Enhancements
        eval_every=args.eval_every,
        eval_start=args.eval_start,
        patience=args.patience,
        stop_threshold=args.stop_threshold,
        classifier_dir=args.classifier_dir,
        l1sp_weight=args.l1sp_weight,
        l2sp_weight=args.l2sp_weight,
        inverse_ewc_lambda=args.inverse_ewc_lambda,
        inverse_ewc_use_l2=args.inverse_ewc_use_l2,
        previous_fisher_path=args.previous_fisher_path,
        save_fisher_path=args.save_fisher_path,
        trajectory_lambda=args.trajectory_lambda,
        previous_delta_path=args.previous_delta_path,
        save_delta_path=args.save_delta_path,
        set_original_params_to_base=args.set_original_params_to_base,
        selft_loss=args.selft_loss,
        selft_topk=args.selft_topk,
        selft_anchor=args.selft_anchor,
        selft_grad_dict_path=args.selft_grad_dict_path,
        selft_mask_dict_path=args.selft_mask_dict_path,
        with_gradient_projection=args.with_gradient_projection,
        gradient_projection_prompts=args.gradient_projection_prompts,
        gradient_projection_num_prompts=args.gradient_projection_num_prompts,
        previously_unlearned=args.previously_unlearned,
    )

    save_state_dict = {}
    for name, param in unet.named_parameters():
        if name in train_param_names:
            save_state_dict[name] = param.cpu().detach().clone()
    torch.save(save_state_dict, args.output_path)
    print(f"Saved EraseDiff UNet to '{args.output_path}'")