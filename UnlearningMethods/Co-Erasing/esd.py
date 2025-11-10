# Standard Library Imports
import os 
import sys
import random
import argparse

# Third-Party Imports
import torch
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler

# Local Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

## Utils
from esd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

## Continual Enhancements
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss
from ContinualEnhancements.SelFT.selft_utils import (
    get_selft_mask_dict,
    apply_selft_masks,
)
from ContinualEnhancements.Projection.gradient_projection import (
    generate_gradient_projection_prompts,
    get_anchor_embeddings,
    apply_gradient_projection,
)

def get_torch_dtype(torch_dtype_string):
    if torch_dtype_string == 'float32':
        return torch.float32
    elif torch_dtype_string == 'bfloat16':
        return torch.bfloat16
    elif torch_dtype_string == 'float16':
        return torch.float16
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype_string}")

def load_sd_models(basemodel_id, torch_dtype, device='cuda:0'):
    print(f"\nLoading Stable Diffusion Model from {basemodel_id} with torch dtype {torch_dtype} on device {device}")
    
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
    print(F"Training method: '{train_method}'")
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
    parser.add_argument('--concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--concept_type', help='type of concept erasure', type=str, required=True, choices=['style', 'object', 'celebrity'])
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--erase_from_prompts', help='Path to text file with erase-from prompts', type=str, required=False, default=None)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)

    # Inference
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3.0)
    
    # Training
    parser.add_argument('--train_method', help='Parameter Update Group', type=str, required=True, choices=['esd-x', 'esd-u', 'esd-all', 'esd-x-strict'])
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=None)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--base_model_dir', help='Base model to use', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--unet_ckpt_path', help='Path to continual unet checkpoint', type=str, default=None)
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    parser.add_argument('--torch_dtype', help='torch dtype to use', type=str, required=False, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--seed', help='Random seed', type=int, required=False, default=42)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model at save path', default=False)
    
    # Continual Enhancements
    parser.add_argument('--l1sp_weight', type=float, default=0.0, help='Weight for L1SP regularizer')
    parser.add_argument('--l2sp_weight', type=float, default=0.0, help='Weight for L2SP regularizer')
    parser.add_argument('--selft_loss', type=str, default=None, choices=['esd', 'ca'], help='Type of SelFT importance loss to use')
    parser.add_argument('--selft_topk', type=float, default=0.01, help='Top-k fraction of parameters to keep for SelFT')
    parser.add_argument('--selft_anchor', type=str, default="", help='Anchor concept used for SelFT when required')
    parser.add_argument('--selft_grad_dict_path', type=str, default=None, help='Path to save/load SelFT gradient dictionary')
    parser.add_argument('--selft_mask_dict_path', type=str, default=None, help='Path to save/load SelFT mask dictionary')
    parser.add_argument('--with_gradient_projection', action='store_true', default=False, help='Enable gradient projection to preserve anchor concepts')
    parser.add_argument('--gradient_projection_prompts', type=str, default=None, help='Path to anchor prompts txt for gradient projection (generated if missing)')
    parser.add_argument('--gradient_projection_num_prompts', type=int, default=400, help='Number of prompts to generate when building anchor set')
    parser.add_argument('--previously_unlearned', type=str, default=None, help='Previously unlearned concepts for anchor prompt generation')

    args = parser.parse_args()

    # Erasing
    erase_concept = args.concept
    concept_type = args.concept_type
    erase_concept_from = args.erase_from
    erase_from_prompts_path = args.erase_from_prompts
    negative_guidance = args.negative_guidance

    # Support text files for concept prompts
    erase_concept_prompts_path = None
    erase_concept_prompts_list = []
    if erase_concept.lower().endswith('.txt'):
        erase_concept_prompts_path = os.path.abspath(os.path.expanduser(erase_concept))
        if not os.path.exists(erase_concept_prompts_path):
            raise FileNotFoundError(f"Concept prompts file not found: {erase_concept_prompts_path}")
        with open(erase_concept_prompts_path, 'r', encoding='utf-8') as handle:
            erase_concept_prompts_list = [line.strip() for line in handle if line.strip()]
        if not erase_concept_prompts_list:
            raise ValueError(f"No prompts found in concept file: {erase_concept_prompts_path}")
    else:
        erase_concept_prompts_list = [erase_concept]

    if erase_concept_prompts_path is not None:
        print(f"Loaded {len(erase_concept_prompts_list)} concept prompt(s) from '{erase_concept_prompts_path}'. Prompts will be randomly sampled during training.")

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
    torch_dtype = get_torch_dtype(args.torch_dtype)
    seed = args.seed
    save_path = args.save_path
    save_parent_dir = os.path.dirname(save_path)
    os.makedirs(save_parent_dir, exist_ok=True)
    base_model_dir = args.base_model_dir
    unet_ckpt_path = args.unet_ckpt_path
    
    # Load erase-from prompts if provided
    erase_from_prompts_list = []
    if erase_from_prompts_path is not None:
        erase_from_prompts_path = os.path.abspath(os.path.expanduser(erase_from_prompts_path))
        if not os.path.exists(erase_from_prompts_path):
            raise FileNotFoundError(f"Erase-from prompts file not found: {erase_from_prompts_path}")
        with open(erase_from_prompts_path, 'r', encoding='utf-8') as handle:
            erase_from_prompts_list = [line.strip() for line in handle if line.strip()]
        if not erase_from_prompts_list:
            raise ValueError(f"No prompts found in erase-from file: {erase_from_prompts_path}")
        if erase_concept_from is not None:
            print("Warning: both --erase_from and --erase_from_prompts provided; using prompts from file.")
    elif erase_concept_from is not None:
        erase_from_prompts_list = [erase_concept_from]

    # Exit if model exists
    if os.path.exists(save_path):
        if not args.overwrite:
            print(f"Model already exists at {save_path}. Exiting to avoid overwriting.")
            sys.exit(0)
        else:
            print(f"Model already exists at {save_path}. Overwriting.")

    # Continual Enhancements
    l1sp_weight = args.l1sp_weight
    l2sp_weight = args.l2sp_weight

    # Loss function
    criteria = torch.nn.MSELoss()

    # Load Model Components and Ckpt if provided
    pipe, base_unet, esd_unet = load_sd_models(basemodel_id=base_model_dir, torch_dtype=torch_dtype, device=device)
    if unet_ckpt_path is not None:
        if not os.path.exists(unet_ckpt_path):
            print(f"UNet checkpoint not found at '{unet_ckpt_path}'. Using default UNet from pipeline '{base_model_dir}'...")
        else:
            state_dict = torch.load(unet_ckpt_path, map_location="cpu")
            base_missing, base_unexpected = base_unet.load_state_dict(state_dict, strict=False)
            esd_missing, esd_unexpected = esd_unet.load_state_dict(state_dict, strict=False)
            print(f"Loaded Continual UNet Ckpt from {unet_ckpt_path}")
            if base_missing or base_unexpected or esd_missing or esd_unexpected:
                print(f"\tBase UNet missing {len(base_missing)} / unexpected {len(base_unexpected)}")
                print(f"\tESD UNet missing {len(esd_missing)} / unexpected {len(esd_unexpected)}")
    
    # Set Progress Bar and Scheduler
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    # SelFT setup
    grad_hooks = []
    selft_mask_dict = None
    if args.selft_loss is not None:
        print(f"Using SelFT with loss '{args.selft_loss}' and top-k '{args.selft_topk}'")
        esd_unet.eval()
        selft_mask_dict = get_selft_mask_dict(
            esd_unet,
            pipe.text_encoder,
            pipe.tokenizer,
            args.selft_mask_dict_path,
            args.selft_grad_dict_path,
            erase_concept_prompts_list,
            args.selft_anchor,
            args.selft_topk,
            args.selft_loss,
            device
        )
        grad_hooks = apply_selft_masks(esd_unet, selft_mask_dict)
        esd_unet.train()

    # Gradient projection setup
    anchor_embeddings_matrix = None
    if args.with_gradient_projection:
        print("\nUsing gradient projection to preserve anchor concepts.")
        anchor_prompts = []
        prompts_path = None
        if args.gradient_projection_prompts:
            prompts_path = os.path.abspath(os.path.expanduser(args.gradient_projection_prompts))
            if os.path.isfile(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as handle:
                    prompts = [line.strip() for line in handle if line.strip()]
                if prompts:
                    print(f"\tLoaded '{len(prompts)}' anchor prompts from '{prompts_path}'")
                    anchor_prompts.extend(prompts)
                else:
                    print(f"\tWarning - '{prompts_path}' is empty. Regenerating anchor prompts.")
            if not anchor_prompts:
                print(f"Generating gradient projection prompts and saving to file: '{prompts_path}'")
                anchor_prompts = generate_gradient_projection_prompts(
                    file_path=prompts_path,
                    num_prompts=args.gradient_projection_num_prompts,
                    concept_type=concept_type,
                    previously_unlearned=args.previously_unlearned,
                    target_concept_list=erase_concept_prompts_list.copy()
                )
        else:
            print("\tWarning - no anchor prompts path provided; gradient projection disabled.")
        
        if anchor_prompts:
            anchor_embeddings_matrix = get_anchor_embeddings(
                anchor_prompts,
                pipe.text_encoder,
                pipe.tokenizer,
                device
            )
            print(f"Gradient projection anchor matrix shape: {anchor_embeddings_matrix.shape}")
        else:
            print("\tNo anchor prompts available; disabling gradient projection.")
            args.with_gradient_projection = False

    # Continual Enhancements
    original_params = {}

    # Set default LR if not provided
    if lr is None:
        if concept_type in ['style', 'celebrity']: lr = 1e-5
        if concept_type in ['object']: lr = 5e-6
        print(f"\nUsing default LR of '{lr:.1e}' for '{concept_type}' unlearning")
    else:
        print(f"\nUsing provided LR of '{lr:.1e}' for '{concept_type}' unlearning")
    print(f"Using negative guidance of '{negative_guidance}' and start guidance scale of '{guidance_scale}'")

    # Setup training
    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeding RNGs with '{seed}'")

    # Get text embeddings we'll use for erasing
    if erase_from_prompts_path is not None:
        print(f"Loaded {len(erase_from_prompts_list)} erase-from prompt(s) from '{erase_from_prompts_path}'")

    if erase_concept_prompts_path is not None:
        concept_prompt_display = f"file '{erase_concept_prompts_path}' ({len(erase_concept_prompts_list)} prompt(s))"
    elif len(erase_concept_prompts_list) > 1:
        concept_prompt_display = f"{len(erase_concept_prompts_list)} concept prompt(s)"
    else:
        concept_prompt_display = f"'{erase_concept_prompts_list[0]}'"

    if erase_from_prompts_list:
        if erase_from_prompts_path is None and len(erase_from_prompts_list) == 1:
            erase_from_display = f"'{erase_from_prompts_list[0]}'"
        elif erase_from_prompts_path is None:
            erase_from_display = f"{len(erase_from_prompts_list)} custom prompt(s)"
        else:
            erase_from_display = f"file '{erase_from_prompts_path}' ({len(erase_from_prompts_list)} prompt(s))"
        print(f"Erasing concept {concept_prompt_display} from {erase_from_display}")
    else:
        print(f"Erasing concept {concept_prompt_display} from itself")

    erase_concept_embeds_cache = []
    erase_from_embeds_cache = []
    use_erase_from = len(erase_from_prompts_list) > 0

    with torch.no_grad():
        # Get text embeddings for concept prompts and empty string
        concept_prompt_input = erase_concept_prompts_list if len(erase_concept_prompts_list) > 1 else erase_concept_prompts_list[0]
        concept_negative_prompt = [""] * len(erase_concept_prompts_list) if isinstance(concept_prompt_input, list) else ""
        erase_concept_embeds, null_embeds = pipe.encode_prompt(prompt=concept_prompt_input,
                                                               device=device,
                                                               num_images_per_prompt=batchsize,
                                                               do_classifier_free_guidance=True,
                                                               negative_prompt=concept_negative_prompt)
        erase_concept_embeds = erase_concept_embeds.to(device)
        null_embeds = null_embeds.to(device)
        for idx in range(len(erase_concept_prompts_list)):
            start = idx * batchsize
            end = start + batchsize
            erase_concept_embeds_cache.append(erase_concept_embeds[start:end])
        null_embeds = null_embeds[:batchsize]
        
        # Get timestep conditioning if needed
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=torch_dtype)
        
        # Get text embeddings for erase-from prompts if provided
        if use_erase_from:
            erase_from_embeds, _ = pipe.encode_prompt(prompt=erase_from_prompts_list,
                                                      device=device,
                                                      num_images_per_prompt=batchsize,
                                                      do_classifier_free_guidance=False,
                                                      negative_prompt="")
            erase_from_embeds = erase_from_embeds.to(device)
            for idx in range(len(erase_from_prompts_list)):
                start = idx * batchsize
                end = start + batchsize
                erase_from_embeds_cache.append(erase_from_embeds[start:end])
    
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

        # Select concept prompt (supports sampling when provided via txt file)
        concept_selection_idx = random.randrange(len(erase_concept_prompts_list))
        selected_concept_prompt = erase_concept_prompts_list[concept_selection_idx]
        selected_concept_embed = erase_concept_embeds_cache[concept_selection_idx]

        # Select erase-from prompt if available
        if use_erase_from:
            erase_from_selection_idx = random.randrange(len(erase_from_prompts_list))
            selected_prompt = erase_from_prompts_list[erase_from_selection_idx]
            selected_erase_from_embed = erase_from_embeds_cache[erase_from_selection_idx]
        else:
            selected_prompt = selected_concept_prompt
            selected_erase_from_embed = None
        display_prompt = selected_prompt if len(selected_prompt) <= 48 else selected_prompt[:45] + '...'
        display_concept_prompt = selected_concept_prompt if len(selected_concept_prompt) <= 48 else selected_concept_prompt[:45] + '...'

        # Generate latent image
        with torch.no_grad():
            # Generate latent image of concept to erase (or erase from if provided)
            xt = pipe(selected_prompt,
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
                encoder_hidden_states=selected_concept_embed,
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
            if use_erase_from:
                noise_pred_erase_from = pipe.unet(
                    xt,
                    run_till_timestep_scheduler,
                    encoder_hidden_states=selected_erase_from_embed,
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
            encoder_hidden_states=selected_concept_embed if not use_erase_from else selected_erase_from_embed,
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

        # Apply gradient projection if enabled
        if args.with_gradient_projection and anchor_embeddings_matrix is not None:
            apply_gradient_projection(
                model=esd_unet,
                filtered_embedding_matrix=anchor_embeddings_matrix,
                device=device,
            )

        # Store losses and update progress bar
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,
                         erase=f"'{display_concept_prompt}'",
                         prompt=f"'{display_prompt}'")

        # Take optimizer step
        optimizer.step()
    
    # Clean up SelFT hooks if registered
    if grad_hooks:
        for hook in grad_hooks:
            hook.remove()
        print(f"Removed {len(grad_hooks)} SelFT gradient hook(s)")

    # Save updated ESD UNet parameters
    print(f"\nSaving full UNet checkpoint to '{save_path}'")
    esd_state_dict = {
        name: param.detach().to("cpu")
        for name, param in esd_unet.state_dict().items()
    }
    if ".safetensors" in save_path:
        save_file(esd_state_dict, save_path)
    else:
        torch.save(esd_state_dict, save_path)
