# Standard Library
import logging
from pathlib import Path
import json

# Third Party
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo


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
        print(vars(args))
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