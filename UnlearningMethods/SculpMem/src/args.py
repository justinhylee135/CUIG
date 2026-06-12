import argparse

def parse_args(input_args=None):
    """
    Argument parser for concept unlearning training script (train_sculpmem.py)
    """
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # Custom Continual Enhancements Arguments
    ## Simultaneous 
    parser.add_argument('--eval_interval', type=int, default=None, help='Evaluate every n iterations')
    parser.add_argument('--patience', type=int, default=2000, help='Patience for early stopping')
    parser.add_argument('--stop_threshold', type=int, default=99.0, help='Unlearn Accuracy threshold to trigger early stopping')
    parser.add_argument('--eval_classifier_dir', type=str, required=False, help='Directory of classifier')
    parser.add_argument('--eval_prompt_dir', type=str, default=None, help='Directory of evaluation prompts')
    ## Weight
    parser.add_argument('--l2sp_weight', type=float, default=0.0, help='Weight for L2SP regularizer')
    parser.add_argument('--l1sp_weight', type=float, default=0.0, help='Weight for L1SP regularizer')
    ## SelFT
    parser.add_argument('--with_selft', action='store_true', help='Whether to apply selective finetuning (gradient masking).')
    parser.add_argument('--selft_mask_dict_path', type=str, default=None, help='Path to save/load mask dictionary')
    parser.add_argument('--selft_grad_dict_path', type=str, default=None, help='Path to save/load gradient dictionary')
    parser.add_argument('--selft_loss', type=str, default='ConAbl', help='Type of importance loss to use')
    parser.add_argument('--selft_topk', type=float, default=0.01, help='Top-k percentage of of parameters by importance.')
    parser.add_argument('--selft_dynamic', action='store_true', default=False, help='Enable SculpMem dynamic attention masking.')
    parser.add_argument('--selft_dynamic_warmup_steps', type=int, default=50, help='Optimization steps before the first dynamic mask update.')
    parser.add_argument('--selft_dynamic_mask_update_interval', type=int, default=100, help='Optimization steps between dynamic mask updates.')
    parser.add_argument('--selft_dynamic_initial_turnover_fraction', type=float, default=0.2, help='Fraction of active masked weights to rotate at each update.')
    parser.add_argument('--selft_dynamic_decay_masks', action='store_true', default=True, help='Use cosine decay for dynamic mask turnover.')
    parser.add_argument('--selft_dynamic_update_mask', action='store_true', default=True, help='Keep updating dynamic masks after warmup.')
    ## Gradient Projection
    parser.add_argument(
        '--with_gradient_projection',
        action='store_true',
        help='Whether to apply gradient projection to preserve anchor concepts.'
    )
    parser.add_argument(
        "--auxiliary_prompts_path",
        type=str,
        default=None,
        help="Path to a file containing auxiliary prompts for gradient projection"
    )
    parser.add_argument(
        "--gradient_projection_num_prompts",
        type=int,
        default=200,
        help="Number of prompts to generate for gradient projection"
    )
    parser.add_argument(
        "--gradient_projection_no_dual_domain",
        action='store_true',
        help='Only generate one domain (style or object)',
        default=False
    )
    parser.add_argument(
        "--gradient_projection_llm_model_id",
        type=str,
        default="mistral",
        help="LLM Model to use for auxiliary prompt generation",
    )
    
    # Custom Training Arguments
    parser.add_argument(
        '--with_style_replacement',
        help='Replace style in prompt rather than appending',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--previously_unlearned',
        help='List of previously unlearned concepts for negative prompting',
        default=None
    )
    parser.add_argument(
        '--overwrite_existing_ckpt', 
        help='Overwrite existing checkpoint if it exists', 
        action='store_true', 
        default=False
    )
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        default=None,
        help="Path to previous timestep UNet checkpoint.",
    )
    parser.add_argument(
        "--turn_on_checkpointing",
        action="store_true",
        help=(
            "Whether to turn on checkpointing. If set, the training state will be saved every `checkpointing_steps`."
            " This is useful for resuming training in case of interruptions."
        ),
        default=False
    )
    # Default Arguments Below
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=["style", "object", "celeb", "character", "nudity", "inappropriate_content"],
        help="the type of removed concepts",
    )
    parser.add_argument(
        "--anchor_target_concepts",
        "--target_concepts",
        dest="anchor_target_concepts",
        type=str,
        required=True,
        help="Target concepts to unlearn.",
    )
    parser.add_argument(
        "--prompt_gen_model",
        type=str,
        default="openai",
        choices=["openai", "meta-llama"],
        help="the type of model to generate anchor prompts",
    )
    parser.add_argument(
        "--instance_anchor_image_paths_file",
        type=str,
        default=None,
        help="Path to a text file containing instance image paths.",
    )
    parser.add_argument(
        "--anchor_dataset_dirs",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_anchor_prompts_file",
        type=str,
        help="Path to a text file containing instance prompts aligned to instance image paths.",
    )
    parser.add_argument(
        "--anchor_prompt_paths",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_anchor_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--anchor_preservation_weight",
        type=float,
        default=1.0,
        help="The weight of anchor preservation loss.",
    )
    parser.add_argument(
        "--num_anchor_images",
        type=int,
        default=200,
        help=(
            "Minimal anchor class images. If there are not enough images already present in"
            " anchor_dataset_dirs, additional images will be sampled with anchor_prompt_paths."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_anchor_prompts",
        type=int,
        default=200,
        help=("Minimal prompts used to generate anchor class images"),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all images in the training dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--anchor_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the anchor dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.0e-06,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default="kv-xattn",
        choices=["full", "xattn", "kv-xattn", "text-emb"],
        help="Parameter groups to set to trainable.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="The token to use to push to Huggingface Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="fp16",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concept_configs",
        "--concepts_list",
        dest="concept_configs",
        type=str,
        default=None,
        help="Path to json containing multiple concept configs, will overwrite parameters like instance_anchor_prompts_file, anchor_prompt_paths, etc.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--hflip", action="store_true", help="Apply horizontal flip data augmentation."
    )
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Check arguments for target to anchor mapping
    if args.with_anchor_preservation:
        if args.concept_configs is None:
            if args.anchor_dataset_dirs is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.anchor_prompt_paths is None:
                raise ValueError("You must specify prompt for class images.")

    return args
