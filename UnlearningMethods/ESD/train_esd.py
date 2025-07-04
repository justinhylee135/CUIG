# Standard Library
import argparse
import random
from pathlib import Path
import os
import sys

# Third Party
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything

# Local
## Utils
from utils import (
    process_input_concepts,
    get_pipelines,
    get_trainable_params,
    sample_model,
    encode_text,
)
## Continual Enhancements
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
### Simultaneous
from ContinualEnhancements.Simultaneous.sim_utils import sample_and_evaluate_ua, check_early_stopping
### Regularization
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss
### SelFT
from ContinualEnhancements.SelFT.selft_utils import (
    get_selft_mask_dict,
    apply_selft_masks
)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method with Diffusers')
    
    # Erasing
    parser.add_argument('--concept', type=str, help="The concept(s) you want to erase", required=True)
    parser.add_argument('--concept_type', type=str, required=True, choices=['style', 'object', 'celebrity'], help='Type of concept to unlearn')
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=1.0)
    
    # Inference
    parser.add_argument('--start_guidance', help='Guidance of initial image', type=float, required=False, default=3)
    parser.add_argument('--ddim_steps', help='DDIM steps of inference', type=int, required=False, default=50)
    parser.add_argument('--image_size', help='Image resolution', type=int, required=False, default=512)
    
    # Training
    parser.add_argument('--train_method', help='Parameters to update', type=str, required=True, choices=['xattn','noxattn', 'selfattn', 'full'])
    parser.add_argument('--iterations', help='Number of iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='Learning rate', type=str, required=False, default=None)
    parser.add_argument('--devices', help='CUDA devices used for loading frozen and esd unet', type=str, required=False, default='0,0')
    parser.add_argument('--seed', help='Random seed for reproducibility', type=int, required=False, default=42)
    parser.add_argument('--sample_latent_from_frz_pipeline', help='Sample latent from frozen pipeline instead of esd pipeline.', action='store_true', default=False)
    
    # Input/Output
    parser.add_argument('--base_model_dir', help='Directory to diffusers pipeline', type=str, required=False, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--save_path', help='output directory to save results', type=str, required=True)
    parser.add_argument('--unet_ckpt', help='Path to UNet checkpoint to load for continual unlearning', type=str, required=False, default=None)
    parser.add_argument('--use_base_for_frz_unet', help='Use base model (instead of unet_ckpt) for frozen UNet', action='store_true', default=False)
    parser.add_argument('--overwrite_existing_ckpt', help='Overwrite existing checkpoint if it exists', action='store_true', default=False)
    parser.add_argument('--verbose', help='Print verbose output', action='store_true', default=False)
    
    # Continual Enhancements
    ## Simultaneous
    parser.add_argument('--eval_every', type=int, default=None, help='Evaluate every n iterations')
    parser.add_argument('--patience', type=int, default=2000, help='Patience for early stopping')
    parser.add_argument('--classifier_dir', type=str, required=False, help='Directory of classifier')
    
    ## Regularizers
    parser.add_argument('--l2sp_weight', type=float, default=0.0, help='Weight for L2SP regularizer')
    parser.add_argument('--l1sp_weight', type=float, default=0.0, help='Weight for L1SP regularizer')
    
    ## SelFT
    parser.add_argument('--selft_loss', type=str, default=None, choices=['esd', 'ca'], help='Type of importance loss to use')
    parser.add_argument('--selft_topk', type=float, default=0.001, help='Top-k percentage of of parameters by importance.')
    parser.add_argument('--selft_anchor', type=str, default="", help='Anchor concept for ca loss')
    parser.add_argument('--selft_grad_dict_path', type=str, default=None, help='Path to save/load gradient dictionary')
    parser.add_argument('--selft_mask_dict_path', type=str, default=None, help='Path to save/load mask dictionary')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Define save path
    save_path = args.save_path
    parent_dir = Path(save_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWill save model to '{save_path}'")
    
    # Exit if model exists
    if os.path.exists(save_path):
        if args.overwrite_existing_ckpt:
            print(f"Overwriting existing model at {save_path}.")
        else:
            print(f"Model already exists at '{save_path}'. Delete existing model to train again.")
            sys.exit(0)

    # Process concept(s) to erase
    prompt_list = process_input_concepts(args.concept, args.concept_type)
    print(f"Prompt list for Unlearning '{args.concept_type}':")
    for i, prompt in enumerate(prompt_list): print(f"{i+1}. '{prompt}'")
    prompt_count = {prompt: 0 for prompt in prompt_list}
     
    # Load pipelines
    args.devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    frz_pipeline, esd_pipeline = get_pipelines(args.base_model_dir, args.unet_ckpt, args.use_base_for_frz_unet, args.devices)
    original_params = {}

    # Choose parameters to train based on train_method
    train_param_names, train_param_values = get_trainable_params(esd_pipeline, args.train_method)
    print(f"Train Method '{args.train_method}': '{len(train_param_names)}' Updatable Parameters")
    i = 1
    for name, param in esd_pipeline.unet.named_parameters():
        if name in train_param_names:
            if args.verbose: print(f"{i}. {name}")
            param.requires_grad = True
            i+=1
        else:
            param.requires_grad = False

    # Set up training parameters
    lr = args.lr
    if lr is None:
        # LR used in our paper
        if args.concept_type in ['style', 'celebrity']: lr = 1e-5
        if args.concept_type in ['object']: lr = 8e-6
        print(f"\nUsing default LR of '{lr}' for '{args.concept_type}' unlearning")
    else:
        lr = float(lr)
        print(f"\nUsing provided LR of '{lr}' for '{args.concept_type}' unlearning")
    opt = torch.optim.Adam(train_param_values, lr=lr)
    criteria = torch.nn.MSELoss()
    esd_pipeline.scheduler.set_timesteps(args.ddim_steps)
    frz_pipeline.scheduler.set_timesteps(args.ddim_steps)
    print(f"Using negative_guidance: '{args.negative_guidance}' and start_guidance: '{args.start_guidance}'")
    
    # Simultaneous Unlearning (Early Stopping)
    if args.eval_every is not None:
        if args.classifier_dir is None: raise ValueError("Classifier directory must be specified for early stopping evaluation.")
        print(f"Using early stopping with patience {args.patience} and eval_every {args.eval_every} instead of {args.iterations} iterations")
        best_ua = 0.0
        no_improvement_count = 0
    
    # Print Regularization settings (if selected)
    if args.l1sp_weight > 0.0: print(f"Using L1SP regularizer with weight '{args.l1sp_weight}'")
    if args.l2sp_weight > 0.0: print(f"Using L2SP regularizer with weight '{args.l2sp_weight}'")
    
    # Initialize SelFT variables
    selft_device = args.devices[0]
    selft_mask_dict = None
    grad_hooks = []
    if args.selft_loss is not None:
        print(f"Using SelFT with loss type: '{args.selft_loss}', top-k: '{args.selft_topk}'")
        
        # Generate or load SelFT masks (would need to adapt for diffusers UNet)
        selft_mask_dict = get_selft_mask_dict(
            esd_pipeline, 
            args.selft_mask_dict_path, 
            args.selft_grad_dict_path, 
            prompt_list, 
            args.selft_anchor, 
            args.selft_topk, 
            args.selft_loss, 
            selft_device
        )
        
        # Apply SelFT masks via gradient hooks
        grad_hooks = apply_selft_masks(esd_pipeline.unet, selft_mask_dict)
        print(f"Applied SelFT masks with '{len(grad_hooks)}' hooks")
    
    # Start Training
    esd_pipeline.unet.train()
    stop_training = False
    iteration = 0
    with tqdm(total=args.iterations, desc="Training", unit="iteration") as pbar:
        while not stop_training:
            # Select unlearning prompt for this iteration
            prompt = random.sample(prompt_list, 1)[0]
            prompt_count[prompt] += 1
            
            # Reset optimizer gradients
            opt.zero_grad()

            # Choose random timestep for sampling
            t_enc = torch.randint(args.ddim_steps, (1,), device=args.devices[0])
            timestep_idx = int(t_enc)
            timestep = esd_pipeline.scheduler.timesteps[timestep_idx]

            # Generate random noise
            latents = torch.randn((1, 4, args.image_size // 8, args.image_size // 8)).to(args.devices[0])
            latents = latents * esd_pipeline.scheduler.init_noise_sigma

            with torch.no_grad():
                # Partially generate latent with the concept to unlearn
                z = sample_model(
                    (frz_pipeline if args.sample_latent_from_frz_pipeline else esd_pipeline), 
                    prompt=prompt, 
                    height=args.image_size, 
                    width=args.image_size,
                    num_inference_steps=args.ddim_steps, 
                    guidance_scale=args.start_guidance, 
                    latents=latents,
                    t_until=timestep_idx
                )
                
                # Get text embeddings
                emb_prompt = encode_text(frz_pipeline.text_encoder, frz_pipeline.tokenizer, prompt, args.devices[1])
                emb_null = encode_text(frz_pipeline.text_encoder, frz_pipeline.tokenizer, "", args.devices[1])
                
                # Prepare inputs for frozen model
                z_input = z.to(args.devices[1])
                timestep_tensor = timestep.unsqueeze(0).to(args.devices[1])
                
                # Get noise predictions from frozen model
                pnoise_frz_null = frz_pipeline.unet(z_input, timestep_tensor, encoder_hidden_states=emb_null).sample
                pnoise_frz_prompt = frz_pipeline.unet(z_input, timestep_tensor, encoder_hidden_states=emb_prompt).sample

            # Prepare input for ESD model
            emb_prompt_esd = encode_text(esd_pipeline.text_encoder, esd_pipeline.tokenizer, prompt, args.devices[0])
            z_input_esd = z.to(args.devices[0])
            timestep_tensor_esd = timestep.unsqueeze(0).to(args.devices[0])
            
            # Get noise predictions from ESD model
            pnoise_prompt_esd = esd_pipeline.unet(z_input_esd, timestep_tensor_esd, encoder_hidden_states=emb_prompt_esd).sample
            
            # No gradients for frozen model
            pnoise_frz_null.requires_grad = False
            pnoise_frz_prompt.requires_grad = False
            
            # Unlearning loss for ESD
            target = pnoise_frz_null.to(args.devices[0]) - (args.negative_guidance * (pnoise_frz_prompt.to(args.devices[0]) - pnoise_frz_null.to(args.devices[0])))
            loss = criteria(pnoise_prompt_esd, target)
            
            # Regularizers
            l1sp_loss = None
            l2sp_loss = None
            if args.l1sp_weight > 0.0:
                l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(esd_pipeline.unet, original_params)
                loss += l1sp_loss
            if args.l2sp_weight > 0.0:
                l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(esd_pipeline.unet, original_params)
                loss += l2sp_loss

            # Take Gradient and Optimizer Step
            loss.backward()
            opt.step()
            
            # Log progress (add regularizer losses if they exist)
            pbar_postfix = {"loss": loss.item(), "timestep": timestep_idx, "prompt": f"'{prompt}'"}
            if l1sp_loss is not None: pbar_postfix["l1sp_loss"] = l1sp_loss.item()
            if l2sp_loss is not None: pbar_postfix["l2sp_loss"] = l2sp_loss.item()
            pbar.set_postfix(pbar_postfix)
            
            # Simultaneous Early Stopping
            iteration+=1
            if args.eval_every is not None and iteration % args.eval_every == 0:
                # Get estimated unlearned accuracy
                ua = sample_and_evaluate_ua(esd_pipeline, args.concept_type, iteration, save_path, prompt_list, 
                                            prompt, args.devices[0], args.classifier_dir)
                print(f"Iteration '{iteration}', Unlearned Accuracy: '{ua}'")
                
                # Check for early stopping
                best_ua, no_improvement_count, stop_training = check_early_stopping(
                    ua, best_ua, no_improvement_count, args.eval_every, args.patience
                )
            pbar.update(1)
            
            # End training if iterations reached or early stopping condition met
            if iteration >= args.iterations: stop_training = True        
            
    # Clean up gradient hooks if SelFT was used
    if grad_hooks:
        for hook in grad_hooks:
            hook.remove()
        print(f"Removed '{len(grad_hooks)}' gradient hooks")
    
    # Finalize training
    esd_pipeline.unet.eval()
    print(f"Training complete. Printing prompt sample counts below")
    for prompt, count in prompt_count.items():
        print(f"Word: '{prompt}' Sample Count: {count}")
    
    # Save UNet updated parameters
    save_state_dict = {}
    for name, param in esd_pipeline.unet.named_parameters():
        if name in train_param_names:
            save_state_dict[name] = param.cpu().detach().clone()
    torch.save(save_state_dict, save_path)
    print(f"Saved ESD UNet to '{save_path}'")
