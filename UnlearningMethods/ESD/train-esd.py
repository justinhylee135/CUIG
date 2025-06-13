# Standard Library Imports
import os 
import sys
import random
import argparse

# Third-Party Imports
import torch
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# Local Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

## Utils
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

## Continual Enhancements
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss

def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    # This will be the frozen base UNet model
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    # This will be the trainable ESD UNet model
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)

    # Stable Diffusion Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    # List of actual parameters values to train
    esd_params = []

    # Name of parameters to train
    esd_param_names = []

    # Loop through all modules in the UNet model
    for name, module in esd_unet.named_modules():
        # Check if the module is a Linear or Conv2d layer (or LoRA compatible layers)
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            # Cross-Attention Only
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

            # Everything Except Cross-Attention                    
            if train_method == 'esd-u' and ('attn2' not in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

            # All Parameters                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

            # Only keys and values of Cross-Attention      
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    # Return parameters and parameters names for training
    return esd_param_names, esd_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDv1.4',
                    description = 'Finetuning stable-diffusion to erase the concepts')
    
    # Erasing
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--concept_type', help='type of concept erasure', type=str, required=True, choices=['style', 'object', 'celebrity'])
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)

    # Inference
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3)
    
    # Training
    parser.add_argument('--train_method', help='Parameter Update Group', type=str, required=True, choices=['esd-x', 'esd-u', 'esd-a', 'esd-x-strict'])
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=1000)
    parser.add_argument('--lr', help='Learning rate', type=float, default=None)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--base_model_dir', help='Base model to use', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--continual_unet_ckpt_path', help='Path to continual unet checkpoint', type=str, default=None)
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')

    # Continual Enhancements
    parser.add_argument('--l1sp_weight', type=float, default=0.0, help='Weight for L1SP regularizer')
    parser.add_argument('--l2sp_weight', type=float, default=0.0, help='Weight for L2SP regularizer')

    args = parser.parse_args()

    # Erasing
    erase_concept = args.erase_concept
    concept_type = args.concept_type
    erase_concept_from = args.erase_from
    negative_guidance = args.negative_guidance

    # Inference
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    height=width=1024

    # Training
    train_method=args.train_method
    iterations = args.iterations
    batchsize = 1
    lr = args.lr
    device = args.device
    torch_dtype = torch.bfloat16
    save_path = args.save_path
    save_parent_dir = os.path.dirname(save_path)
    base_model_dir = args.base_model_dir
    continual_unet_ckpt_path = args.continual_unet_ckpt_path
    os.makedirs(save_parent_dir, exist_ok=True)
    
    # Continual Enhancements
    l1sp_weight = args.l1sp_weight
    l2sp_weight = args.l2sp_weight

    # Loss function
    criteria = torch.nn.MSELoss()

    # Load Model Components and Ckpt if provided
    pipe, base_unet, esd_unet = load_sd_models(basemodel_id=base_model_dir, torch_dtype=torch_dtype, device=device)
    if continual_unet_ckpt_path is not None:
        state_dict = torch.load(continual_unet_ckpt_path, map_location='cpu')
        base_unet.load_state_dict(state_dict, strict=False)
        esd_unet.load_state_dict(state_dict, strict=False)
        print(f"Loaded Continual UNet Ckpt from {continual_unet_ckpt_path}")
    
    # Set Progress Bar and Scheduler
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    # Continual Enhancements
    original_params = {}

    # Set default LR if not provided
    if lr is None:
        if concept_type in ['style', 'celebrity']: lr = 1e-5
        if concept_type in ['object']: lr = 5e-6
        print(f"Using default LR of {lr} for {concept_type} unlearning")
        
    # Setup training
    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    # Get text embeddings we'll use for erasing
    with torch.no_grad():
        # Get text embedding for concept to erase and empty string
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')                      
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        # Get timestep conditioning if needed
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=torch_dtype)
        
        # Get text embedding for concept to erase from if provided
        if erase_concept_from is not None:
            erase_from_embeds, _ = pipe.encode_prompt(prompt=erase_concept_from,
                                                                device=device,
                                                                num_images_per_prompt=batchsize,
                                                                do_classifier_free_guidance=False,
                                                                negative_prompt="",
                                                                )
            erase_from_embeds = erase_from_embeds.to(device)
    
    # Set progress bar and store loss
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []

    # START TRAINING
    for iteration in pbar:
        # Clear gradients
        optimizer.zero_grad()

        # Load frozen UNet
        pipe.unet = base_unet

        # Select a random timestep [0, num_inference_steps) to run inference till
        run_till_timestep = random.randint(0, num_inference_steps-1)

        # Get the actual timestep values corresponding to the scheduler
        run_till_timestep_scheduler = pipe.scheduler.timesteps[run_till_timestep]

        # Select random seed
        seed = random.randint(0, 2**15) 

        # Generate latent image
        with torch.no_grad():
            # Generate latent image of concept to erase (or erase from if provided)
            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                  num_images_per_prompt=batchsize,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  run_till_timestep = run_till_timestep,
                  generator=torch.Generator().manual_seed(seed),
                  output_type='latent',
                  height=height,
                  width=width,
                 ).images

            # Get the predicted noise conditioned on concept to erase
            noise_pred_erase = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # Get the predicted noise conditioned on empty string
            noise_pred_null = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # Get predicted noise conditioned on erase_from embedding if provided
            if erase_concept_from is not None:
                noise_pred_erase_from = pipe.unet(
                    xt,
                    run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                # Otherwise just set to predicted noise conditioned on concept to erase
                noise_pred_erase_from = noise_pred_erase
        
        # Now load the trainable UNet
        pipe.unet = esd_unet

        # Get predicted noise conditioned on erase embedding (or erase_from if provided)
        noise_pred_esd_model = pipe.unet(
            xt,
            run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        
        # This loss will change our ESD UNet so that when we try to generate the concept to erase, we'll instead 
        # generate an image corresponding to the negative cfg of the concept to erase.
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        
        # Continual Enhancements
        ## L1SP Regularization
        if l1sp_weight > 0.0:
            loss += l1sp_weight * calculate_l1sp_loss(esd_unet, original_params)
        ## L2SP Regularization
        if l2sp_weight > 0.0:
            loss += l2sp_weight * calculate_l2sp_loss(esd_unet, original_params)

        # Get gradient
        loss.backward()

        # Store losses and update progress bar
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)

        # Take optimizer step
        optimizer.step()
    
    # Save updated ESD UNet parameters
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    if erase_concept_from is None:
        erase_concept_from = erase_concept
    save_file(esd_param_dict, save_path)