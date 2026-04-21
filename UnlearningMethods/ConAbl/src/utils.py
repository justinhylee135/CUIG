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
from tqdm.auto import tqdm
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.utils import is_wandb_available, is_xformers_available
from packaging import version
from huggingface_hub import create_repo, HfApi

# Local/project-specific
from src.model import CustomDiffusionPipeline, freeze_params, write_model_card_to_output_dir

## Continual Enhancements
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)
### SelFT
from Regularizers.SelFT.selft import (
    get_selft_mask_dict,
    apply_selft_masks
)
### Gradient Projection
from Regularizers.Projection.projection import (
    generate_auxiliary_prompts,
    get_auxiliary_embeddings,
    build_orthogonal_projector
)

# Most functions here are code chunks I cut from the original ConAbl main()
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
    logging_dir = Path(args.output_dir, args.logging_dir) # Default logging_dir is {output_dir}/logs
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit) # Default: no limit
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Default: 1
        mixed_precision=args.mixed_precision, # Default: no
        log_with=args.report_to, # Default: None
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    # Handle wandb import if needed
    if args.report_to == "wandb":
        if not is_wandb_available(): raise ImportError("Install wandb if you want to use it for logging.")
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
        accelerator.init_trackers("ConAbl", config=vars(args))
    
    return accelerator, logger

def setup_concepts_list(args):
    """
    Prepare and normalize concept-related argument fields for training/unlearning workflows.
    This helper parses list-like CLI string inputs (e.g., "['a','b']"), expands singleton
    values to match the number of concepts to unlearn, and optionally broadcasts shorter
    `anchor_dataset_dirs` / `anchor_prompt_paths` lists when their lengths evenly divide
    the number of concepts. It then builds:
    - `args.concept_configs` (if not provided): list of per-concept dict entries containing
        `anchor_prompt_path`, `anchor_dataset_dir`, `anchor_target_concept`,
        `instance_anchor_prompts_file`, and `instance_anchor_image_paths_file`.
    - `args.target_concepts`: derived textual prompts used by continual-enhancement methods,
        based on `args.concept_type`.
    If `args.concept_configs` is provided as a path, the function loads it as JSON instead
    of constructing it programmatically.
    Args:
            args: Namespace-like object expected to define at least:
                    - anchor_target_concepts (str; optionally list literal)
                    - anchor_dataset_dirs (str; optionally list literal)
                    - anchor_prompt_paths (str; optionally list literal)
                    - concept_configs (None or JSON file path)
                    - concept_type (one of: "object", "celeb", "character", "style")
                    - instance_anchor_prompts_file (optional sequence)
                    - instance_anchor_image_paths_file (optional sequence)
    Side Effects:
            Mutates `args` in place:
                    - Ensures `anchor_target_concepts`, `anchor_dataset_dirs`, and
                      `anchor_prompt_paths` are lists.
                    - Sets/overwrites `target_concepts`.
                    - Sets/overwrites `concept_configs` (constructed list or loaded JSON object).
    Raises:
            ValueError: If `args.concept_type` is unsupported when building `target_concepts`.
            (May also propagate parsing/index/file errors from `ast.literal_eval`, indexing,
            or JSON file loading.)
    """
    # Open bracket means number of concepts to unlearn is >= 1
    if "[" in args.anchor_target_concepts:
        # Convert arg to list of strings
        args.anchor_target_concepts = ast.literal_eval(args.anchor_target_concepts)
        num_concepts = len(args.anchor_target_concepts)

        # Check path to directories for anchor dataset
        if "[" in args.anchor_dataset_dirs:
            # Convert to list of directories 
            args.anchor_dataset_dirs = ast.literal_eval(args.anchor_dataset_dirs)

            # Check for mismatch in number of datasets
            num_datasets = len(args.anchor_dataset_dirs)
            if num_datasets < num_concepts:
                # If it's num_datasets | num_concepts then broadcast
                if num_concepts % num_datasets == 0:
                    print(f"Only '{num_datasets}' datasets were provided for '{num_concepts}' concepts. Broadcasting '{num_concepts // num_datasets}' times...")
                    args.anchor_dataset_dirs = args.anchor_dataset_dirs * (num_concepts // num_datasets)
                else:
                    # Return error
                    print("Error: Number of concepts to unlearn is not divisble by number of anchor datasets.")
                    sys.exit(1)
        else:
            # If multiple paths were not given, just broadcast to match number of concepts to unlearn
            args.anchor_dataset_dirs = [args.anchor_dataset_dirs] * num_concepts
        
        # Check paths to anchor prompts
        if "[" in args.anchor_prompt_paths:
            args.anchor_prompt_paths = ast.literal_eval(args.anchor_prompt_paths)

            # Check for mismatch in number of prompts
            num_prompts = len(args.anchor_prompt_paths)
            if num_prompts < num_concepts:
                if num_concepts % num_prompts == 0:
                    print(f"Only '{num_prompts}' prompts were provided for '{num_concepts}' concepts. Broadcasting '{num_concepts // num_prompts}' times...")
                    args.anchor_prompt_paths = args.anchor_prompt_paths * (num_concepts // num_prompts)
                else:
                    # Return error
                    print("Error: Number of concepts to unlearn is not divisble by number of anchor datasets.")
                    sys.exit(1)
        else:
            # If multiple paths were not given, just broadcast to match number of concepts to unlearn
            args.anchor_prompt_paths = [args.anchor_prompt_paths] * num_concepts
    else:
        # We're only unlearning one concept
        args.anchor_target_concepts = [args.anchor_target_concepts]
        args.anchor_dataset_dirs = [args.anchor_dataset_dirs]
        args.anchor_prompt_paths = [args.anchor_prompt_paths]

    if args.concept_configs is None:
        num_concepts = len(args.anchor_target_concepts)
        args.concept_configs = []

        for i in range(num_concepts):
            # Create concept_configs entry
            args.concept_configs.append({
                "anchor_prompt_path": args.anchor_prompt_paths[i], # Path to anchor prompts txt file
                "anchor_dataset_dir": args.anchor_dataset_dirs[i], # Directory to anchor dataset
                "anchor_target_concept": args.anchor_target_concepts[i], # The concept to unlearn
                "instance_anchor_prompts_file": args.instance_anchor_prompts_file[i] if args.instance_anchor_prompts_file else None, # Default empty
                "instance_anchor_image_paths_file": args.instance_anchor_image_paths_file[i] if args.instance_anchor_image_paths_file else None, # Default empty
            })
    else:
        if os.path.exists(args.concept_configs):
            print(f"Loading json concept configs from '{args.concept_configs}'...")
            with open(args.concept_configs, "r") as f:
                args.concept_configs = json.load(f)
        else:
            print(f"ERROR: '{args.concept_configs}' is not a path to a json file. To auto-generate it, leave this argument as None.")
            sys.exit(1)

    # Some regularizers require list of target and anchor concepts.
    args.target_concepts = []
    args.anchor_concepts = []
    for concept in args.concept_configs:
        anchor_target_concept = concept["anchor_target_concept"]
        if args.concept_type in ["object", "celeb", "character"]:
            # These concepts allow users to specify an anchor concept:
            # "anchor+target". We use only the target side for derived prompts.
            args.target_concepts.append(anchor_target_concept.split("+")[1])
            args.anchor_concepts.append(anchor_target_concept.split("+")[0])
        elif args.concept_type == "style":
            if "+" in anchor_target_concept:
                print(f"ERROR: '{anchor_target_concept}' includes '+' for concept type 'style'.")
                sys.exit(1)
            args.target_concepts.append(anchor_target_concept.replace("_", " ") + " Style")
            args.anchor_concepts.append("")
        else:
            raise ValueError(f"Unknown concept type: '{args.concept_type}'. Supported types are 'object' and 'style' for target_concepts.")

def setup_hf_repo(args):
    """
    Set up Hugging Face Hub repository if push_to_hub is enabled.
    
    Args:
        args: Arguments object containing hub configuration parameters
        
    Returns:
        str or None: Repository ID if pushing to hub, None otherwise
    """
    # Default: Don't save to huggingface
    if not args.push_to_hub:
        return None
    
    # Select repo name
    repo_name = args.hub_model_id or Path(args.output_dir).name
    print(f"Creating repository: {repo_name}")
    
    # Create huggingface repo to save to
    repo_id = create_repo(
        repo_id=repo_name,
        exist_ok=True,
        token=args.hf_token,
    )
    print(f"Repository created/found: '{repo_id}'")
    
    # Return the hub_model_id
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

    # Enable xformers if requested (I personally used it)
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

    # Enable gradient checkpointing if requested (Default: False)
    if args.gradient_checkpointing:
        print(f"Enabling gradient checkpointing...")
        unet.enable_gradient_checkpointing()
        if args.parameter_group == "text-emb":
            text_encoder.gradient_checkpointing_enable()
            
    # Enable TF32 for faster training on Ampere GPUs (Default: False)
    if args.allow_tf32:
        print(f"Enabling TF32 for faster training on Ampere GPUs...")
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate (I personally used it)
    if args.scale_lr:
        # Default base learning rate
        old_lr = args.learning_rate

        # Scale learning rate by number of gradient acc steps (Default: 1), train batch size (Default: 4), and num processes (Default: 1 GPU)
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.anchor_batch_size
            * accelerator.num_processes
        )

        # Double learning rate if using anchor preservation (default: False)
        if args.with_anchor_preservation: # Diffusion training loss on anchor concept
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
        tuple: (optimizer, token_ids_to_unlearn, params_to_optimize)
    """
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs (Default: False)
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

    # Handle tokens ids to unlearn when unlearning via text embeddings
    token_ids_to_unlearn = []
    if args.parameter_group == "text-emb":
        print(f"Setting up text embedding unlearning...")

        # Tokenize each concept to unlearn
        for t_concept in args.target_concepts:
            # Convert the target concept to token ids
            t_concept_ids = tokenizer.encode(t_concept, add_special_tokens=False)
            print(f"Tokenized '{t_concept}' to '{t_concept_ids}'")
            
            # Add list to tokens to unlearn
            token_ids_to_unlearn += t_concept_ids

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
        # We already set trainable UNet parameters in model/create_custom_unet()
        trainable_named = [(n, p) for n, p in unet.named_parameters() if p.requires_grad]
        params_to_optimize = (p for _, p in trainable_named)
        print(f"UNet Parameter Group: '{args.parameter_group}' with '{len(trainable_named)}' keys")

    # Create optimizer
    print(f"Initializing training configurations...")
    print(f"\tCreating optimizer with class '{optimizer_class.__name__}'")
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), # Default: (0.9, 0.999)
        weight_decay=args.adam_weight_decay, # Default: 0.01
        eps=args.adam_epsilon, # Default: 1e-08
    )

    return optimizer, token_ids_to_unlearn

def setup_regularizers(args, unet, text_encoder, tokenizer, device):
    """
    Set up Regularizers for Continual Unlearning.
    """
    # Regularizers
    print(f"Setting up Regularizers for Continual Unlearning...")

    ## Anchor Preservation 
    if args.with_anchor_preservation: 
        print(f"\tUsing Anchor Preservation loss with weight '{args.anchor_preservation_weight}'")
    else:
        print("\tNot using Anchor Preservation loss, only Unlearning loss will be computed.")

    ## Simultaneous Unlearning (Early Stopping)
    if args.eval_interval is not None:
        # The iterations argument now serves as a maximum number of iterations rather than fixed
        print(f"\tUsing early stopping with patience '{args.patience}' and evaluating every '{args.eval_interval}' iterations with maximum of '{args.iterations}' iterations")
        
        # Store the best unlearning accuracy
        args.best_ua = 0.0

        # Store number of iterations without any improvements to unlearning accuracy
        args.no_improvement_count = 0

        # Whether early stopping has been triggered
        args.early_stop_triggered = False

    ## Weight Regularization
    # The parameters we will regularize towards (leave empty until first iteration)
    args.original_params = {}

    # Print status message
    if args.l1sp_weight > 0.0: print(f"\tUsing L1SP regularizer with weight '{args.l1sp_weight}'")
    if args.l2sp_weight > 0.0: print(f"\tUsing L2SP regularizer with weight '{args.l2sp_weight}'")

    ## Selective Fine-tuning (SelFT)
    # The device to store the gradient mask
    selft_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # This will hold the gradient mask
    selft_mask_dict = None

    # We will apply gradient masks as gradient hooks (function triggered whenever we backprop)
    grad_hooks = []

    # User can select which loss we use to calculcate parameter importance
    if args.with_selft:
        print(f"\tUsing SelFT with loss type: '{args.selft_loss}', top-k: '{args.selft_topk}'")
        unet.eval()

        # Check if save/load path for mask dictionary exists
        if args.selft_mask_dict_path is None:
            print(f"\t'args.selft_mask_dict_path' is none, defaulting to 'args.output_dir/mask_dict.pt': '{args.output_dir}/mask_dict.pt'")
            args.selft_mask_dict_path = f"{args.output_dir}/mask_dict.pt"

        # Check if save/load path for gradient dictionary exists        
        if args.selft_grad_dict_path is None:
            print(f"\t'args.selft_grad_dict_path' is none, defaulting to 'args.output_dir/grad_dict.pt': '{args.output_dir}/grad_dict.pt'")
            args.selft_grad_dict_path = f"{args.output_dir}/grad_dict.pt"

        # Create gradient masks
        selft_mask_dict = get_selft_mask_dict(
            unet, 
            text_encoder, 
            tokenizer,
            args.selft_mask_dict_path, # Path to save/load gradient mask
            args.selft_grad_dict_path, # Path to save/load parameter importance dictionary
            args.target_concepts, # Concepts to unlearn
            args.anchor_concepts, # Anchor concepts to map to
            args.selft_topk, # Top k% of parameters to update
            args.selft_loss, # Loss type to calculate parameter importance
            selft_device # Device to store gradient masks
        )

        # Apply SelFT masks via gradient hooks
        grad_hooks = apply_selft_masks(unet, selft_mask_dict)
        print(f"\t\tApplied SelFT masks with '{len(grad_hooks)}' hooks")
        
    ## Gradient Projection
    if args.with_gradient_projection:
        print(f"\tUsing Regularizer: Gradient Projection")

        # Store generated or retrieved auxiliary prompts
        auxiliary_prompts_list = []

        # If path to auxiliary prompts is given
        if args.auxiliary_prompts_path:
            # This was the option used for Object Unlearning in our paper

            # Load auxiliary prompts
            if os.path.isfile(args.auxiliary_prompts_path):
                print(f"\t\tLoading Auxiliary Prompts from file: '{args.auxiliary_prompts_path}'")
                with open(args.auxiliary_prompts_path, 'r') as f:
                    auxiliary_prompts_list = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # Generate Auxiliary prompts
                print(f"\t\tGenerating Auxiliary Prompts and saving to file: '{args.auxiliary_prompts_path}'")
                auxiliary_prompts_list = generate_auxiliary_prompts(
                    auxiliary_prompts_path=args.auxiliary_prompts_path,
                    num_prompts=args.gradient_projection_num_prompts,
                    concept_type=args.concept_type,
                    previously_unlearned=args.previously_unlearned, # Default: None
                    target_concepts=args.target_concepts,
                    dual_domain=(not args.gradient_projection_no_dual_domain), # Default: Yes to dual_domain
                    llm_model_id=args.gradient_projection_llm_model_id
                )
        else:
            # This was the option used for Style Unlearning in our paper
            print(f"\t\tCollecting Auxiliary Prompts from Anchor Prompts...")

            # Iterate through each concept cfg
            for i, concept in enumerate(args.concept_configs):

                # Get anchor prompt file path
                anchor_prompt_file_path = concept["anchor_prompt_path"]

                # Extract anchor prompts
                if os.path.isfile(anchor_prompt_file_path):
                    with open(anchor_prompt_file_path, 'r') as f:
                        prompts = [line.strip() for line in f.readlines() if line.strip()]
                    print(f"\t\t\tConcept {i+1}: Loaded '{len(prompts)}' anchor prompts from '{anchor_prompt_file_path}'")
                    auxiliary_prompts_list.extend(prompts)
                else:
                    print(f"\t\t\tConcept {i+1}: Warning - anchor_prompt_path file '{anchor_prompt_file_path}' not found")
        
        # Status
        print(f"\t\tTotal Auxiliary Prompts collected: '{len(auxiliary_prompts_list)}'")

        # Create embedding matrix
        auxiliary_embeddings_matrix = get_auxiliary_embeddings(
            auxiliary_prompts_list, text_encoder, tokenizer, device
        )

        # Create orthogonal projector to axuiliary embeddings
        args.aux_orth_proj = build_orthogonal_projector(auxiliary_embeddings_matrix, device) 

def prepare_training_state(args, accelerator, anchor_dataloader):
    """
    Compute training-loop bookkeeping and optional resume state.

    Args:
        args: Arguments object
        accelerator: Accelerator object
        anchor_dataloader: Dataloader used for training

    Returns:
        tuple: (completed_opt_steps, num_completed_epochs, progress_bar, num_completed_iterations_in_epoch)
            num_completed_opt_steps: completed optimizer-update count
            num_completed_epochs: epoch index to start/resume from
            num_completed_iterations_in_epoch: number of iterations to skip after skipping completed epochs
            progress_bar: tqdm progress bar over optimizer updates
    """

    # Number of optimizer updates in one epoch of the anchor dataloader.
    num_opt_steps_per_epoch = math.ceil(len(anchor_dataloader) / args.gradient_accumulation_steps)

    # In `setup_data_and_scheduler()`, `args.iterations` may have been derived from `args.epochs`. 
    # Recompute it only in that case because dataloader length can change after `accelerator.prepare(...)`.
    if args._overrode_iterations: args.iterations = args.epochs * num_opt_steps_per_epoch

    # Keep epochs consistent after the final dataloader length is known.
    args.epochs = math.ceil(args.iterations / num_opt_steps_per_epoch)

    # Completed optimizer-update step count.
    num_completed_opt_steps = 0

    # Number of completed epochs
    num_completed_epochs = 0

    # Number of iterations after the last epoch to resume from
    num_completed_iterations_in_epoch = 0

    # Optionally restore a previous training state from checkpoint.
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: '{args.resume_from_checkpoint}'")
        
        # User provided an explicit checkpoint path.
        if args.resume_from_checkpoint != "latest":
            ckpt_dir = os.path.basename(args.resume_from_checkpoint)
        else:
            # Otherwise, search `output_dir` for the newest `checkpoint-<step>`.
            candidate_dirs = os.listdir(args.output_dir)

            # Search for directories that start with "checkpoint"
            candidate_dirs = [
                d for d in candidate_dirs if d.startswith("checkpoint")
            ]

            # Sort directories by number in directory name (number of opt steps)
            candidate_dirs = sorted(
                candidate_dirs, key=lambda x: int(x.split("-")[1])
            )

            # Set ckpt directory to the candidate dir with the highest number
            ckpt_dir = (
                candidate_dirs[-1] if len(candidate_dirs) > 0 else None
            )

        # If no checkpoint exists, continue as a fresh training run.
        if ckpt_dir is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new optimization run...")
            args.resume_from_checkpoint = None
        else:
            # Restore model/optimizer/scheduler/accelerator state.
            accelerator.print(f"Resuming from checkpoint '{ckpt_dir}'")
            accelerator.load_state(os.path.join(args.output_dir, ckpt_dir))

            # `checkpoint-<N>` stores the completed optimizer-update count.
            num_completed_opt_steps = int(ckpt_dir.split("-")[1])

            # Convert optimizer updates to itrations. This assumes the same gradient accumulation setting is used when resuming.
            num_completed_iterations = num_completed_opt_steps * args.gradient_accumulation_steps

            # Compute number of epochs we've already completed
            num_completed_epochs = num_completed_opt_steps // num_opt_steps_per_epoch

            # Number of iterations we need per epoch
            num_iterations_per_epoch = num_opt_steps_per_epoch * args.gradient_accumulation_steps
            
            # This now gives us the number of completd iterations AFTER we complete "num_completed_epochs" epochs
            num_completed_iterations_in_epoch = num_completed_iterations % num_iterations_per_epoch

    # Progress bar tracks optimizer updates (not iterations).
    progress_bar = tqdm(
        range(num_completed_opt_steps, args.iterations),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Opt. Steps")

    return num_completed_opt_steps, num_completed_epochs, num_completed_iterations_in_epoch, progress_bar

def print_training_settings_summary(args, accelerator, anchor_dataloader, anchor_dataset):
    """ Print a summary of the training configuration."""

    # Target and Anchor Concept
    print(f"Unlearning target to anchor mappings:")
    if args.concept_type == "style":
        for i, concept in enumerate(args.concept_configs):
            target_style = f"{concept['anchor_target_concept'].replace('_', ' ')} Style"
            print(f"\t{i+1}. Mapping target style '{target_style}' to anchor prompts in '{concept['anchor_prompt_path']}'.")
    else:
        for i, concept in enumerate(args.concept_configs):
            anchor_concept, target_concept = concept["anchor_target_concept"].split("+")
            if anchor_concept == "*":
                print(f"\t{i+1}. Mapping target concept '{target_concept}' to anchor prompts in '{concept['anchor_prompt_path']}'.")
            else:
                print(f"\t{i+1}. Mapping target concept '{target_concept}' to anchor concept'{anchor_concept}' via string replacement using prompts '{concept['anchor_prompt_path']}'.")

    # Compute actual batch size in optimization step
    total_batch_size = (
        args.anchor_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Final message
    print("***** Running training *****")
    print(f"  Num anchor examples = {len(anchor_dataset)}")
    print(f"  Num batches each epoch = {len(anchor_dataloader)}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Instantaneous batch size per device = {args.anchor_batch_size}")
    print(f"  Total batch size in one optimization step (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.iterations}")

def isolate_target_concept_emb_gradients(
    args, accelerator, text_encoder, tokenizer, token_ids_to_unlearn
):
    """
    Zero out gradients for all token embeddings except the target concept tokens.

    Args:
        args: Training arguments (must have `parameter_group`)
        accelerator: HuggingFace `accelerate` accelerator object
        text_encoder: The text encoder module (e.g., CLIPTextModel)
        tokenizer: The tokenizer used to convert prompt text to input IDs
        token_ids_to_unlearn: List of token IDs that should retain their gradients
    """
    # Validity check
    if args.parameter_group != "text-emb":
        raise ValueError("This function should only be called when unlearning via text encoder")

    # Get gradient of token embeddings (handle DDP or not)
    grads_text_encoder = (
        text_encoder.module.get_input_embeddings().weight.grad
        if accelerator.num_processes > 1
        else text_encoder.get_input_embeddings().weight.grad
    )

    # Create a tensor of all token indices
    all_token_indices = torch.arange(len(tokenizer), device=grads_text_encoder.device)

    # Initialize a boolean mask set to True for all tokens
    # We'll use this to mark which tokens should have their gradients zeroed
    index_of_gradients_to_zero_out = torch.ones_like(all_token_indices, dtype=torch.bool)
    
    # For each token ID we want to preserve, set its mask value to False
    # This ensures we keep gradients only for the target concept tokens
    for token_id_to_unlearn in token_ids_to_unlearn:
        # If the index is equal to the a token id we're going to unlearn, then set the mask to 0 for that token
        index_of_gradients_to_zero_out &= all_token_indices != token_id_to_unlearn

    # Zero out gradients for non target concept embeddings
    grads_text_encoder.data[index_of_gradients_to_zero_out, :] = 0

def get_params_to_clip(parameter_group, text_encoder, unet):
    """
    Select parameters for gradient clipping.

    Args:
        parameter_group: Trainable parameter group
        text_encoder: The text encoder model
        unet: The UNet model

    Returns:
        An iterable of trainable model parameters (`requires_grad=True`)
    """

    # If we're unlearning via text encoder
    if parameter_group == "text-emb":
        # Clip all trainable text encoder parameters
        return (param for param in text_encoder.parameters() if param.requires_grad)

    # For all UNet-based parameter groups, clip only trainable parmas.
    return (param for param in unet.parameters() if param.requires_grad)

def update_progress_and_checkpoint(
    accelerator, progress_bar, num_completed_opt_steps, args,
    unet, text_encoder, tokenizer, token_ids_to_unlearn
):
    """
    Update progress bar and save checkpoint if needed when gradients are synced.
    
    Args:
        accelerator: Accelerator object
        progress_bar: Progress bar to update
        num_completed_opt_steps: Number of optimization steps we've completed
        args: Arguments object with checkpointing settings
        logger: Logger object
        unet: UNet model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        token_ids_to_unlearn: Modifier token IDs
        
    Returns:
        int: Updated global step
    """

    # Ensure gradients are synced
    if accelerator.sync_gradients:

        # Update progress bar (records number of optimization steps)
        progress_bar.update(1)
        num_completed_opt_steps += 1

        # If we're storing checkpoints (Default: False). Save every "checkpointing_steps" steps (default: 250)
        if args.turn_on_checkpointing and (num_completed_opt_steps % args.checkpointing_steps == 0):

            # Have only the main process handle saving
            if accelerator.is_main_process:
                # Load our current unlearning unet (or text encoder) into a diffusion pipeline
                diffusion_pipeline = CustomDiffusionPipeline.from_pretrained(
                    args.base_model_dir,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.hf_revision,
                    modifier_token_id=token_ids_to_unlearn,
                )

                # Set save path for ckpt
                ckpt_save_path = os.path.join(args.output_dir, f"delta-{num_completed_opt_steps}")

                # Save the updated parameters
                diffusion_pipeline.save_pretrained(ckpt_save_path, parameter_group=args.parameter_group)

                # Status
                print(f"Saved ckpt at step '{num_completed_opt_steps}' to '{ckpt_save_path}'")

    # Return updated number of completed optimization steps
    return num_completed_opt_steps


def upload_to_hf(args, repo_id):
    """
    Upload model and results to Hugging Face Hub.
    
    Args:
        args: Arguments object
        repo_id: Repository ID
        images: Optional list of validation images
    """
    # Save model card
    write_model_card_to_output_dir(
        repo_id,
        base_model_name=args.base_model_dir,
        anchor_target_concepts=args.anchor_target_concepts,
        repo_folder=args.output_dir,
    )
    
    # Upload to hub
    api = HfApi(token=args.hf_token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        path_in_repo=".",
        repo_type="model",
    )
