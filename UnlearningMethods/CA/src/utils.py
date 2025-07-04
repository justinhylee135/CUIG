# Standard Library
import logging
from pathlib import Path
import json
import os
import math
import itertools

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

# All functions here are code chunks cut from main
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
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
                "caption_target": args.caption_target,
            }
        ]
    else:
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
        print(f"Setting up UNet parameter group: '{args.parameter_group}'")
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

    # Create optimizer
    print(f"Creating optimizer with class '{optimizer_class.__name__}'")
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return optimizer, modifier_token_id, params_to_optimize

def prepare_training_state(args, accelerator, logger, unet, text_encoder, optimizer, train_dataloader, lr_scheduler):
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

    # Training information
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

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

    # Handle checkpoint resumption
    global_step = 0
    first_epoch = 0
    resume_step = 0

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

    # Load UNet checkpoint for continual unlearning
    if args.unet_ckpt:
        print(f"Loading UNet checkpoint from '{args.unet_ckpt}'...")
        accelerator.load_state(args.unet_ckpt)
        print(f"Loaded UNet checkpoint from '{args.unet_ckpt}'")

    # Set up progress bar
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    return prepared_components, global_step, first_epoch, progress_bar, resume_step


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
            images = _run_final_validation(
                args, accelerator, pipeline, final_epoch
            )

        # Upload to hub if requested
        if args.push_to_hub and repo_id:
            _upload_to_hub(args, repo_id, images)


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