# Standard Library
import logging
from pathlib import Path
import json
import os
import math
import itertools
import ast
import sys

# Third Party
import torch
import numpy as np
from tqdm.auto import tqdm
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.utils import is_wandb_available, is_xformers_available
from diffusers.schedulers import DPMSolverMultistepScheduler
from packaging import version
from huggingface_hub import create_repo, HfApi

# Local/project-specific
from src.model import CustomDiffusionPipeline, freeze_params, save_model_card

## Continual Enhancements
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)
### Regularization
from ContinualEnhancements.Regularization.inverse_ewc import (  
    load_fisher_information,
)
from ContinualEnhancements.Regularization.trajectory import (
    load_delta_from_path,
    save_delta_to_path
)
### SelFT
from ContinualEnhancements.SelFT.selft_utils import (
    get_selft_mask_dict,
    apply_selft_masks
)
### Gradient Projection
from ContinualEnhancements.Projection.gradient_projection import (
    get_anchor_embeddings,
    generate_gradient_projection_prompts
)

# Most functions here are code chunks cut from main
def check_for_existing_ckpt(args):
    """
    Check if the output directory already contains a checkpoint.
    
    Args:
        args: Arguments object with output directory
        
    Raises:
        FileExistsError: If checkpoint already exists and overwrite_existing_ckpt is False
    """
    if args.output_dir is not None: 
        os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "delta.bin")
    if os.path.exists(save_path):
        if args.overwrite_existing_ckpt:
            print(f"Checkpoint '{save_path}' already exists. Overwriting it as per the argument '--overwrite_existing_ckpt'.")
        else:
            print(f"Model already exists at '{save_path}'. Set flag '--overwrite_existing_ckpt' to overwrite it.")
            sys.exit(0)

def setup_accelerator_and_logging(args):
    """
    Set up Accelerator and logging configuration.
    
    Args:
        args: Arguments object containing configuration parameters
        
    Returns:
        tuple: (accelerator, logger) - Configured accelerator and logger objects
    """
    # Set up Logging and Accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    # Handle wandb import if needed
    if args.report_to == "wandb":
        if not is_wandb_available(): 
            raise ImportError("Install wandb if you want to use it for logging.")
        import wandb
    
    # Configure basic logging (one log per process)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    print(f"\nSetting up logger...")
    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)
    
    # Configure transformers and diffusers logging levels
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Initialize trackers on main process
    if accelerator.is_main_process:
        accelerator.init_trackers("concept-ablation", config=vars(args))
    
    return accelerator, logger

def setup_concepts_list(args):
    # Load in lists of concepts, class prompts, and data directories
    if "[" in args.caption_target:
        args.caption_target = ast.literal_eval(args.caption_target)
        num_concepts = len(args.caption_target)
        if "[" in args.class_data_dir:
            args.class_data_dir = ast.literal_eval(args.class_data_dir)
            # Check for mismatch in number of datasets
            num_datasets = len(args.class_data_dir)
            if num_datasets < num_concepts and num_concepts % num_datasets == 0:
                print(f"Only '{num_datasets}' datasets were provided for '{num_concepts}' concepts. Broadcasting '{num_concepts // num_datasets}' times...")
                args.class_data_dir = args.class_data_dir * (num_concepts // num_datasets)
        else:
            args.class_data_dir = [args.class_data_dir] * num_concepts
        if "[" in args.class_prompt:
            args.class_prompt = ast.literal_eval(args.class_prompt)
            # Check for mismatch in number of prompts
            num_prompts = len(args.class_prompt)
            if num_prompts < num_concepts and num_concepts % num_prompts == 0:
                print(f"Only '{num_prompts}' prompts were provided for '{num_concepts}' concepts. Broadcasting '{num_concepts // num_prompts}' times...")
                args.class_prompt = args.class_prompt * (num_concepts // num_prompts)
        else:
            args.class_prompt = [args.class_prompt] * num_concepts
    else:
        args.caption_target = [args.caption_target]
        args.class_data_dir = [args.class_data_dir]
        args.class_prompt = [args.class_prompt]
    
    # We'll also build prompt_list for continual enhancements methods
    args.prompt_list = []

    if args.concepts_list is None:
        num_concepts = len(args.caption_target)
        args.concepts_list = []
        for i in range(num_concepts):
            # Create concepts_list entry
            args.concepts_list.append({
                "class_prompt": args.class_prompt[i],
                "class_data_dir": args.class_data_dir[i],
                "caption_target": args.caption_target[i],
                "instance_prompt": args.instance_prompt[i] if args.instance_prompt else None, # Default empty
                "instance_data_dir": args.instance_data_dir[i] if args.instance_data_dir else None, # Default empty
            })
            
            # Create prompt_list entry
            if args.concept_type == "object":
                if "+" in args.caption_target[i]:
                    args.prompt_list.append(args.caption_target[i].split("+")[1])
                else:
                    args.prompt_list.append(args.caption_target[i])
            elif args.concept_type == "style":
                args.prompt_list.append(args.caption_target[i].replace("_", " ") + " Style")
            else:
                raise ValueError(f"Unknown concept type: '{args.concept_type}'. Supported types are 'object' and 'style' for prompt_list.")
    else:
        print(f"Loading json concepts list from '{args.concepts_list}'...")
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)


def setup_hub_repository(args):
    """
    Set up Hugging Face Hub repository if push_to_hub is enabled.
    
    Args:
        args: Arguments object containing hub configuration parameters
        
    Returns:
        str or None: Repository ID if pushing to hub, None otherwise
    """
    if not args.push_to_hub:
        return None
        
    repo_name = args.hub_model_id or Path(args.output_dir).name
    print(f"Creating repository: {repo_name}")
    
    repo_id = create_repo(
        repo_id=repo_name,
        exist_ok=True,
        token=args.hub_token,
    )
    print(f"Repository created/found: {repo_id}")
    
    # Return the hub_model_id for consistency with original behavior
    return args.hub_model_id

def setup_training_configuration(args, unet, text_encoder, accelerator, logger, weight_dtype):
    """
    Configure training settings like mixed precision, xformers, gradient checkpointing, etc.
    
    Args:
        args: Arguments object
        unet: UNet model
        text_encoder: Text encoder model
        accelerator: Accelerator object
        logger: Logger object
        weight_dtype: Weight data type for mixed precision
    """
    # Move models to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        print(f"Enabling xFormers memory efficient attention...")
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # if args.unet_ckpt:
    #     if not os.path.exists(args.unet_ckpt):
    #         print(f"UNet checkpoint not found at '{args.unet_ckpt}'. Using default UNet from pipeline '{args.base_model_dir}'...")
    #     else:
    #         print(f"Loading UNet checkpoint from '{args.unet_ckpt}'...")
    #         ckpt_sd = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)
    #         if "unet" in ckpt_sd: ckpt_sd = ckpt_sd["unet"]
    #         missing, unexpected = unet.load_state_dict(ckpt_sd, strict=False)
    #         print(f"Loaded '{len(ckpt_sd)}' keys from UNet checkpoint with '{len(missing)}' missing and '{len(unexpected)}' unexpected keys.")
            
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print(f"Enabling gradient checkpointing...")
        unet.enable_gradient_checkpointing()
        if args.parameter_group == "embedding":
            text_encoder.gradient_checkpointing_enable()
            
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        print(f"Enabling TF32 for faster training on Ampere GPUs...")
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate
    if args.scale_lr:
        old_lr = args.learning_rate
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0
        print(f"Scaled learning rate from '{old_lr}' to '{args.learning_rate}'")
    else:
        print(f"Using learning rate: '{args.learning_rate}'")

def setup_optimizer_and_params(args, unet, text_encoder, tokenizer):
    """
    Set up optimizer and parameters to optimize.
    
    Args:
        args: Arguments object
        unet: UNet model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        
    Returns:
        tuple: (optimizer, modifier_token_id, params_to_optimize)
    """
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        print(f"Using 8-bit Adam optimizer...")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Handle modifier tokens for embedding fine-tuning
    modifier_token_id = []
    if args.parameter_group == "embedding":
        print(f"Setting up text embedding finetuning...")
        assert (
            args.concept_type != "memorization"
        ), "embedding finetuning is not supported for memorization"

        for concept in args.concepts_list:
            # Convert the caption_target to ids
            token_ids = tokenizer.encode(
                [concept["caption_target"]], add_special_tokens=False
            )
            print(token_ids)
        # Check if initializer_token is a single token or a sequence of tokens
        modifier_token_id += token_ids

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
        params_to_optimize = itertools.chain(
            text_encoder.get_input_embeddings().parameters()
        )
    else:
        # Set up parameters based on parameter group
        if args.parameter_group == "cross-attn":
            params_to_optimize = itertools.chain(
                [
                    x[1]
                    for x in unet.named_parameters()
                    if ("attn2.to_k" in x[0] or "attn2.to_v" in x[0])
                ]
            )
        elif args.parameter_group == "attn":
            params_to_optimize = itertools.chain(
                [
                    x[1]
                    for x in unet.named_parameters()
                    if ("to_q" in x[0] or "to_k" in x[0] or "to_v" in x[0] or "to_out" in x[0])
                ]
            )
        elif args.parameter_group == "full-weight":
            params_to_optimize = itertools.chain(unet.parameters())
        params_to_optimize, params_copy = itertools.tee(params_to_optimize)
        num_params = sum(1 for _ in params_copy)
        print(f"UNet Parameter Group: '{args.parameter_group}' with '{num_params}' keys")

    # Create optimizer
    print(f"Initializing training configurations...")
    print(f"\tCreating optimizer with class '{optimizer_class.__name__}'")
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return optimizer, modifier_token_id, params_to_optimize

def setup_continual_enhancement(args, unet, text_encoder, tokenizer, device):
    """
    Set up continual enhancement components.
    """
    print(f"Setting up Continual Enhancements...")
    # Continual Enhancements
    ## Retention Loss
    if args.with_prior_preservation: 
        print(f"\tUsing retention loss with weight '{args.prior_loss_weight}'")
    else:
        print("\tNot using retention loss, only unlearning loss will be computed.")

    ## Simultaneous Unlearning (Early Stopping)
    if args.eval_every is not None:
        if args.classifier_dir is None: raise ValueError("Classifier directory must be specified for early stopping evaluation.")
        print(f"\tUsing early stopping with patience '{args.patience}' and eval_every '{args.eval_every}' instead of '{args.max_train_steps}' iterations")
        args.best_ua = 0.0
        args.no_improvement_count = 0
        args.stop_training = False

    ## Regularization    
    if args.set_original_params_to_base:
        print(f"\targs.original_params have already been initialized to the base model")
    else:
        args.original_params = {}
    if args.l1sp_weight > 0.0: print(f"\tUsing L1SP regularizer with weight '{args.l1sp_weight}'")
    if args.l2sp_weight > 0.0: print(f"\tUsing L2SP regularizer with weight '{args.l2sp_weight}'")
    ## Inverse EWC
    args.current_fisher = {}
    args.previous_aggregated_fisher = None
    if args.inverse_ewc_lambda > 0.0:
        print(f"\tUsing Inverse EWC regularizer with lambda '{args.inverse_ewc_lambda}'")
        
        # Choose parameter difference metric
        if args.inverse_ewc_use_l2:
            print(f"\t\tUsing L2 distance for Inverse EWC loss")
        else:
            print(f"\t\tUsing L1 distance for Inverse EWC loss")
        
        # Load previous fisher information
        if args.previous_fisher_path is not None and os.path.exists(args.previous_fisher_path):
            print(f"\t\tLoading previous fisher information from '{args.previous_fisher_path}'")
            args.previous_aggregated_fisher = load_fisher_information(args.previous_fisher_path, device)
        else:
            args.inverse_ewc_lambda = 0.0
            print(f"\t\tNo previous fisher information found. Turning off Inverse EWC regularization...")
        
        if args.save_fisher_path is not None:
            print(f"\t\tWill accumulate fisher information during unlearning and save to '{args.save_fisher_path}'")
    ## Trajectory Regularization
    args.previous_aggregated_delta = None
    if args.trajectory_lambda > 0.0:
        print(f"\tUsing Trajectory regularizer with lambda '{args.trajectory_lambda}'")

        if args.previous_delta_path is not None and os.path.exists(args.previous_delta_path):
            print(f"\t\tLoading previous parameter deltas from '{args.previous_delta_path}'")
            args.previous_aggregated_delta = load_delta_from_path(args.previous_delta_path, device)
        else:
            print(f"\t\tNo previous parameter deltas found. Turning off Trajectory regularization...")
            args.trajectory_lambda = 0.0
    ## SelFT
    selft_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selft_mask_dict = None
    grad_hooks = []
    if args.selft_loss is not None:
        print(f"\tUsing SelFT with loss type: '{args.selft_loss}', top-k: '{args.selft_topk}'")
        # Generate or load SelFT masks (would need to adapt for diffusers UNet)
        unet.eval()
        selft_mask_dict = get_selft_mask_dict(
            unet, text_encoder, tokenizer,
            args.selft_mask_dict_path, args.selft_grad_dict_path, 
            args.prompt_list, args.selft_anchor, 
            args.selft_topk, args.selft_loss, selft_device
        )
        # Apply SelFT masks via gradient hooks
        grad_hooks = apply_selft_masks(unet, selft_mask_dict)
        print(f"\t\tApplied SelFT masks with '{len(grad_hooks)}' hooks")
    ## Gradient Projection
    if args.with_gradient_projection:
        print(f"\tUsing gradient projection to preserve anchor concepts.")
        args.anchor_prompts = []
        if args.gradient_projection_prompts:
            if os.path.isfile(args.gradient_projection_prompts):
                print(f"\t\tLoading anchor prompts from file: '{args.gradient_projection_prompts}'")
                with open(args.gradient_projection_prompts, 'r') as f:
                    args.anchor_prompts = [line.strip() for line in f.readlines() if line.strip()]
            else:
                print(f"\t\tGenerating gradient projection prompts and saving to file: '{args.gradient_projection_prompts}'")
                args.anchor_prompts = generate_gradient_projection_prompts(
                    file_path=args.gradient_projection_prompts,
                    num_prompts=args.gradient_projection_num_prompts,
                    concept_type=args.concept_type,
                    previously_unlearned=args.previously_unlearned,
                    target_concept_list=args.prompt_list,
                    dual_domain=(not args.gradient_projection_no_dual_domain)
                )
        else:
            print(f"\t\tCollecting anchor prompts from concepts_list...")
            for i, concept in enumerate(args.concepts_list):
                class_prompt_file = concept["class_prompt"]
                if os.path.isfile(class_prompt_file):
                    with open(class_prompt_file, 'r') as f:
                        prompts = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"\t\t\tConcept {i+1}: Loaded '{len(prompts)}' anchor prompts from '{class_prompt_file}'")
                    args.anchor_prompts.extend(prompts)
                else:
                    print(f"\t\t\tConcept {i+1}: Warning - class_prompt file '{class_prompt_file}' not found")
        print(f"\t\tTotal anchor prompts collected: '{len(args.anchor_prompts)}'")
        args.anchor_embeddings_matrix = get_anchor_embeddings(
            args.anchor_prompts, text_encoder, tokenizer, device
        )
        

def prepare_training_state(args, accelerator, unet, text_encoder, optimizer, train_dataloader, lr_scheduler):
    """
    Prepare training state with accelerator and set up logging, checkpoints, etc.
    
    Args:
        args: Arguments object
        accelerator: Accelerator object
        logger: Logger object
        unet: UNet model
        text_encoder: Text encoder model
        optimizer: Optimizer
        train_dataloader: Training dataloader
        lr_scheduler: Learning rate scheduler
        
    Returns:
        tuple: (prepared_components, global_step, first_epoch, progress_bar, resume_step)
    """
    # Prepare everything with accelerator
    print(f"Moving components to accelerator...")
    if args.parameter_group == "embedding":
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
        prepared_components = (text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        prepared_components = (unet, optimizer, train_dataloader, lr_scheduler)

    # Recalculate training steps after preparation
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args._overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Setup step counters
    global_step = 0
    first_epoch = 0
    resume_step = 0

    # Load previous training state if resuming
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: '{args.resume_from_checkpoint}'")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Set up progress bar
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    return prepared_components, global_step, first_epoch, progress_bar, resume_step

def print_training_settings_summary(args, accelerator, logger, train_dataloader):
    """ Print a summary of the training configuration."""
    # Target and Anchor Concept
    print(f"Unlearning targets:")
    if args.concept_type == "object":
        for i, concept in enumerate(args.concepts_list):
            if "+" in concept["caption_target"]:
                anchor, target = concept["caption_target"].split("+")
                if anchor == "*":
                    print(f"\t{i+1}. Mapping target concept '{target}' to full anchor prompts in '{concept['class_prompt']}'.")
                else:
                    print(f"\t{i+1}. Mapping target concept '{target}' to anchor '{anchor}' via replacement using prompts '{concept['class_prompt']}'.")
            else:
                print(f"\t{i+1}. Unlearning target object '{concept['caption_target']}'.")
    elif args.concept_type == "style":
        for i, concept in enumerate(args.concepts_list):
            style = f"{concept['caption_target'].replace('_', ' ')} Style"
            print(f"\t{i+1}. Unlearning target style '{style}'.")

    # Total batch size
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Final message
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(args._train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

def zero_out_embedding_gradients_except_concept(
    args, accelerator, text_encoder, tokenizer, modifier_token_id
):
    """
    Zero out gradients for all token embeddings except the concept modifier tokens.

    This ensures only the newly added concept embeddings are updated during training.

    Args:
        args: Training arguments (must have `parameter_group`)
        accelerator: HuggingFace `accelerate` accelerator object
        text_encoder: The text encoder module (e.g., CLIPTextModel)
        tokenizer: The tokenizer used to convert prompt text to input IDs
        modifier_token_id: List of token IDs that should retain their gradients
    """
    if args.parameter_group != "embedding":
        return  # Only applies when fine-tuning embeddings

    # Get gradient of token embeddings (handle DDP or not)
    grads_text_encoder = (
        text_encoder.module.get_input_embeddings().weight.grad
        if accelerator.num_processes > 1
        else text_encoder.get_input_embeddings().weight.grad
    )

    # Create mask for tokens to zero out (everything except modifier_token_id)
    all_token_indices = torch.arange(len(tokenizer), device=grads_text_encoder.device)
    index_grads_to_zero = torch.ones_like(all_token_indices, dtype=torch.bool)
    for token_id in modifier_token_id:
        index_grads_to_zero &= all_token_indices != token_id

    # Zero out gradients for non-concept embeddings
    grads_text_encoder.data[index_grads_to_zero, :] = 0

def get_params_to_clip(args, text_encoder, unet):
    """
    Selects the parameters to apply gradient clipping to, based on the specified parameter group.

    Args:
        args: Training arguments (must include `parameter_group`)
        text_encoder: The text encoder model
        unet: The UNet model

    Returns:
        An iterable of model parameters to be clipped
    """
    if args.parameter_group == "embedding":
        return itertools.chain(text_encoder.parameters())

    elif args.parameter_group == "cross-attn":
        return itertools.chain(
            [param for name, param in unet.named_parameters() if "attn2" in name]
        )

    else:  # full-weight, attn, or other
        return itertools.chain(unet.parameters())

def update_progress_and_checkpoint(
    accelerator, progress_bar, global_step, args, logger, 
    unet, text_encoder, tokenizer, modifier_token_id
):
    """
    Update progress bar and save checkpoint if needed when gradients are synced.
    
    Args:
        accelerator: Accelerator object
        progress_bar: Progress bar to update
        global_step: Current global step
        args: Arguments object with checkpointing settings
        logger: Logger object
        unet: UNet model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        modifier_token_id: Modifier token IDs
        
    Returns:
        int: Updated global step
    """
    if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

        if args.turn_on_checkpointing and (global_step % args.checkpointing_steps == 0):
            if accelerator.is_main_process:
                pipeline = CustomDiffusionPipeline.from_pretrained(
                    args.base_model_dir,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.revision,
                    modifier_token_id=modifier_token_id,
                )
                save_path = os.path.join(
                    args.output_dir, f"delta-{global_step}"
                )
                pipeline.save_pretrained(
                    save_path, parameter_group=args.parameter_group
                )
                logger.info(f"Saved state to {save_path}")

    return global_step

def run_validation_step(
    accelerator, args, logger, global_step, epoch, 
    unet, text_encoder, tokenizer, modifier_token_id
):
    """
    Run validation step if conditions are met.
    
    Args:
        accelerator: Accelerator object
        args: Arguments object with validation settings
        logger: Logger object
        global_step: Current global step
        epoch: Current epoch
        unet: UNet model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        modifier_token_id: Modifier token IDs
    """
    if accelerator.is_main_process:
        if (
            args.turn_on_validation and
            args.validation_prompt is not None
            and global_step % args.validation_steps == 0
        ):
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}."
            )
            
            # Create pipeline
            pipeline = CustomDiffusionPipeline.from_pretrained(
                args.base_model_dir,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                revision=args.revision,
                modifier_token_id=modifier_token_id,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # Run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed
            )
            images = [
                pipeline(
                    args.validation_prompt,
                    num_inference_steps=25,
                    generator=generator,
                    eta=1.0,
                ).images[0]
                for _ in range(args.num_validation_images)
            ]

            # Log to trackers
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "validation", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    # Import wandb only when needed
                    if is_wandb_available():
                        import wandb
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}"
                                    )
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

            # Cleanup
            del pipeline
            torch.cuda.empty_cache()
            
def finalize_training(
    accelerator, args, unet, text_encoder, tokenizer, 
    modifier_token_id, repo_id, final_epoch=0
):
    """
    Finalize training by saving the model, running final validation, and uploading to hub.
    
    Args:
        accelerator: Accelerator object
        args: Arguments object with training settings
        unet: UNet model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        modifier_token_id: Modifier token IDs
        repo_id: Repository ID for hub upload (can be None)
        final_epoch: Final epoch number for logging (default: 0)
    """
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Convert UNet to float32 and create final pipeline
        unet = unet.to(torch.float32)
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.base_model_dir,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
            modifier_token_id=modifier_token_id,
        )
        
        # Save the final model
        save_path = os.path.join(args.output_dir, "delta.bin")
        pipeline.save_pretrained(save_path, parameter_group=args.parameter_group)

        # Run final validation if enabled
        images = None
        if args.turn_on_validation and args.validation_prompt and args.num_validation_images > 0:
            images = run_final_validation(
                args, accelerator, pipeline, final_epoch
            )

        # Upload to hub if requested
        if args.push_to_hub and repo_id:
            upload_to_hub(args, repo_id, images)


def run_final_validation(args, accelerator, pipeline, final_epoch):
    """
    Run final validation inference and log results.
    
    Args:
        args: Arguments object
        accelerator: Accelerator object
        pipeline: Diffusion pipeline
        final_epoch: Final epoch number for logging
        
    Returns:
        list: Generated images
    """
    # Configure pipeline for inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Generate validation images
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = [
        pipeline(
            args.validation_prompt,
            num_inference_steps=25,
            generator=generator,
            eta=1.0,
        ).images[0]
        for _ in range(args.num_validation_images)
    ]

    # Log final results to trackers
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "test", np_images, final_epoch, dataformats="NHWC"
            )
        if tracker.name == "wandb":
            # Import wandb only when needed
            if is_wandb_available():
                import wandb
                tracker.log(
                    {
                        "test": [
                            wandb.Image(
                                image, caption=f"{i}: {args.validation_prompt}"
                            )
                            for i, image in enumerate(images)
                        ]
                    }
                )
    
    return images

def upload_to_hub(args, repo_id, images=None):
    """
    Upload model and results to Hugging Face Hub.
    
    Args:
        args: Arguments object
        repo_id: Repository ID
        images: Optional list of validation images
    """
    # Save model card
    save_model_card(
        repo_id,
        images=images,
        base_model=args.base_model_dir,
        prompt=args.instance_prompt,
        repo_folder=args.output_dir,
    )
    
    # Upload to hub
    api = HfApi(token=args.hub_token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        path_in_repo=".",
        repo_type="model",
    )