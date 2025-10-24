import warnings
warnings.filterwarnings("ignore")

from argparse import Namespace
import logging
import math
import os
import sys
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from src.mace_lora_atten_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from src.cfr_utils import *
from src.dataset import MACEDataset
import json
from safetensors.torch import load_file as safe_load

# Add ContinualEnhancements imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)
print(f"Project root added to sys.path: {project_root}")

### Simultaneous
from ContinualEnhancements.Simultaneous.sim_utils import sample_and_evaluate_ua, check_early_stopping

### Regularization
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss
from ContinualEnhancements.Regularization.inverse_ewc import (
    accumulate_fisher,
    calculate_inverse_ewc_loss,
    save_fisher_information,
    load_fisher_information
)
from ContinualEnhancements.Regularization.trajectory import (
    calculate_trajectory_loss,
    save_delta_to_path,
    load_delta_from_path
)

### Projection
from ContinualEnhancements.Projection.gradient_projection import (
    apply_gradient_projection,
    get_anchor_embeddings,
    generate_gradient_projection_prompts
)

### SelFT
from ContinualEnhancements.SelFT.selft_utils import (
    get_selft_mask_dict,
    apply_selft_masks
)

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    instance_prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["preserve_prompt_ids"] for example in examples]
        pixel_values += [example["preserve_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if masks[0] is not None:
        ## object/celebrity erasure
        masks = torch.stack(masks)
    else:
        ## artistic style erasure
        masks = None
    
    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "masks": masks,
        "concept_positions": concept_positions,
    }
    return batch


def setup_continual_enhancement(args, unet, text_encoder, tokenizer, device):
    """
    Set up continual enhancement components.
    """
    print(f"Setting up Continual Enhancements...")
    # Continual Enhancements
    ## Retention Loss
    if getattr(args, 'with_prior_preservation', False): 
        print(f"\tUsing retention loss with weight '{getattr(args, 'prior_loss_weight', 1.0)}'")
    else:
        print("\tNot using retention loss, only unlearning loss will be computed.")

    ## Simultaneous Unlearning (Early Stopping)
    if getattr(args, 'eval_every', None) is not None:
        if getattr(args, 'classifier_dir', None) is None: 
            raise ValueError("Classifier directory must be specified for early stopping evaluation.")
        print(f"\tUsing early stopping with patience '{getattr(args, 'patience', 5)}' and eval_every '{args.eval_every}' instead of '{args.max_train_steps}' iterations")
        args.best_ua = 0.0
        args.no_improvement_count = 0
        args.stop_training = False

    ## Regularization    
    args.original_params = {}
    if getattr(args, 'l1sp_weight', 0.0) > 0.0: 
        print(f"\tUsing L1SP regularizer with weight '{args.l1sp_weight}'")
    if getattr(args, 'l2sp_weight', 0.0) > 0.0: 
        print(f"\tUsing L2SP regularizer with weight '{args.l2sp_weight}'")
    
    ## Inverse EWC
    args.current_fisher = {}
    args.previous_aggregated_fisher = None
    if getattr(args, 'inverse_ewc_lambda', 0.0) > 0.0:
        print(f"\tUsing Inverse EWC regularizer with lambda '{args.inverse_ewc_lambda}'")
        
        # Choose parameter difference metric
        if getattr(args, 'inverse_ewc_use_l2', False):
            print(f"\t\tUsing L2 distance for Inverse EWC loss")
        else:
            print(f"\t\tUsing L1 distance for Inverse EWC loss")
        
        # Load previous fisher information
        previous_fisher_path = getattr(args, 'previous_fisher_path', None)
        if previous_fisher_path is not None and os.path.exists(previous_fisher_path):
            print(f"\t\tLoading previous fisher information from '{previous_fisher_path}'")
            args.previous_aggregated_fisher = load_fisher_information(previous_fisher_path, device)
        else:
            args.inverse_ewc_lambda = 0.0
            print(f"\t\tNo previous fisher information found. Turning off Inverse EWC regularization...")
        
        save_fisher_path = getattr(args, 'save_fisher_path', None)
        if save_fisher_path is not None:
            print(f"\t\tWill accumulate fisher information during unlearning and save to '{save_fisher_path}'")
    
    ## Trajectory Regularization
    args.previous_aggregated_delta = None
    if getattr(args, 'trajectory_lambda', 0.0) > 0.0:
        print(f"\tUsing Trajectory regularizer with lambda '{args.trajectory_lambda}'")

        previous_delta_path = getattr(args, 'previous_delta_path', None)
        if previous_delta_path is not None and os.path.exists(previous_delta_path):
            print(f"\t\tLoading previous parameter deltas from '{previous_delta_path}'")
            args.previous_aggregated_delta = load_delta_from_path(previous_delta_path, device)
        else:
            print(f"\t\tNo previous parameter deltas found. Turning off Trajectory regularization...")
            args.trajectory_lambda = 0.0
    
    ## SelFT
    selft_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selft_mask_dict = None
    grad_hooks = []
    if getattr(args, 'selft_loss', None) is not None:
        print(f"\tUsing SelFT with loss type: '{args.selft_loss}', top-k: '{getattr(args, 'selft_topk', 0.1)}'")
        # Generate or load SelFT masks
        unet.eval()
        selft_mask_dict = get_selft_mask_dict(
            unet, text_encoder, tokenizer,
            getattr(args, 'selft_mask_dict_path', None),
            getattr(args, 'selft_grad_dict_path', None), 
            getattr(args, 'prompt_list', []),
            getattr(args, 'selft_anchor', 'object'),
            getattr(args, 'selft_topk', 0.1),
            args.selft_loss,
            selft_device
        )
        # Apply SelFT masks via gradient hooks
        grad_hooks = apply_selft_masks(unet, selft_mask_dict)
        print(f"\t\tApplied SelFT masks with '{len(grad_hooks)}' hooks")
        args.grad_hooks = grad_hooks
    
    ## Gradient Projection
    if getattr(args, 'with_gradient_projection', False):
        print(f"\tUsing gradient projection to preserve anchor concepts.")
        args.anchor_prompts = []
        gradient_projection_prompts = getattr(args, 'gradient_projection_prompts', None)
        
        if gradient_projection_prompts:
            if os.path.isfile(gradient_projection_prompts):
                print(f"\t\tLoading anchor prompts from file: '{gradient_projection_prompts}'")
                with open(gradient_projection_prompts, 'r') as f:
                    args.anchor_prompts = [line.strip() for line in f.readlines() if line.strip()]
            else:
                print(f"\t\tGenerating gradient projection prompts and saving to file: '{gradient_projection_prompts}'")
                args.anchor_prompts = generate_gradient_projection_prompts(
                    file_path=gradient_projection_prompts,
                    num_prompts=getattr(args, 'gradient_projection_num_prompts', 100),
                    concept_type=getattr(args, 'concept_type', 'object'),
                    previously_unlearned=getattr(args, 'previously_unlearned', []),
                    target_concept_list=getattr(args, 'prompt_list', []),
                    dual_domain=(not getattr(args, 'gradient_projection_no_dual_domain', False))
                )
        else:
            print(f"\t\tCollecting anchor prompts from multi_concept...")
            # For MACE, collect anchor prompts from the dataset or multi_concept configuration
            if hasattr(args, 'multi_concept') and args.multi_concept:
                for concept_info in args.multi_concept[0]:
                    if isinstance(concept_info, tuple) and len(concept_info) > 1:
                        concept_name = concept_info[0]
                        print(f"\t\t\tAdding concept: {concept_name}")
                        # You might want to generate prompts for each concept
                        args.anchor_prompts.append(f"a photo of {concept_name}")
        
        print(f"\t\tTotal anchor prompts collected: '{len(args.anchor_prompts)}'")
        if args.anchor_prompts:
            args.anchor_embeddings_matrix = get_anchor_embeddings(
                args.anchor_prompts, text_encoder, tokenizer, device
            )
        else:
            print(f"\t\tWarning: No anchor prompts collected. Disabling gradient projection.")
            args.with_gradient_projection = False
            args.anchor_embeddings_matrix = None


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    if args.unet_ckpt is not None:
        if not os.path.isfile(args.unet_ckpt):
            print(f"Invalid UNet checkpoint path: '{args.unet_ckpt}' defaulting to '{args.pretrained_model_name_or_path}/unet'")
        else:
            print(f"Loading unet weights from {args.unet_ckpt}")
            if args.unet_ckpt.endswith(".safetensors"):
                ckpt_sd = safe_load(args.unet_ckpt)
            else:
                ckpt_sd = torch.load(args.unet_ckpt, map_location='cpu')
            unet.load_state_dict(ckpt_sd, strict=False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"Using weight_dtype: '{weight_dtype}'")
        
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            print(f"Using xformers")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        print(f"Using gradient checkpointing")
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        print("Enabling TF32 for matmul and cuDNN.")
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        og_lr = args.learning_rate
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        print(f"Scaling learning rate from {og_lr} to {args.learning_rate}.")

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        
        optimizer_class = bnb.optim.AdamW8bit
        print("Using 8-bit Adam")
    else:
        optimizer_class = torch.optim.AdamW

    if args.with_prior_preservation:
        print("Using prior preservation")
        args.preservation_info = {
                "preserve_prompt": args.preserve_prompt,
                "preserve_data_dir": args.preserve_data_dir
            }
    else:
        args.preservation_info = None
    
    train_dataset = MACEDataset(
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        use_pooler=args.use_pooler,
        multi_concept=args.multi_concept[0],
        mapping=args.mapping_concept,
        augment=args.augment,
        batch_size=args.train_batch_size,
        with_prior_preservation=args.with_prior_preservation,
        preserve_info=args.preservation_info,
        train_seperate=args.train_seperate,
        aug_length=args.aug_length,
        prompt_len=args.prompt_len,
        input_data_path=args.input_data_dir,
        use_gpt=args.use_gpt,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Setup continual enhancements
    setup_continual_enhancement(args, unet, text_encoder, tokenizer, accelerator.device)

    # stage 1: closed-form refinement
    print(f"Acquiring cross-attention outputs")
    projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
    
    # to save memory
    CFR_dict = {}
    max_concept_num = args.max_memory # the maximum number of concept that can be processed at once
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = None
            CFR_dict[f'{layer_num}_for_mat2'] = None
            
        for i in tqdm(range(0, len(train_dataset.dict_for_close_form), max_concept_num)):
            contexts_sub, valuess_sub = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                                    train_dataset.dict_for_close_form[i:i+5], tokenizer, all_words=args.all_words)
            closed_form_refinement(projection_matrices, contexts_sub, valuess_sub, cache_dict=CFR_dict, cache_mode=True)
            
            del contexts_sub, valuess_sub
            gc.collect()
            torch.cuda.empty_cache()
            
    else:
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = .0
            CFR_dict[f'{layer_num}_for_mat2'] = .0
            
        contexts, valuess = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, 
                                        train_dataset.dict_for_close_form, tokenizer, all_words=args.all_words)
    
    del ca_layers, og_matrices

    # Load cached prior knowledge for preserving
    if args.prior_preservation_cache_path:
        prior_preservation_cache_dict = torch.load(args.prior_preservation_cache_path, map_location=projection_matrices[0].weight.device)
    else:
        prior_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(projection_matrices))):
            prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
            
    # Load cached domain knowledge for preserving
    if args.domain_preservation_cache_path:
        domain_preservation_cache_dict = torch.load(args.domain_preservation_cache_path, map_location=projection_matrices[0].weight.device)
    else:
        domain_preservation_cache_dict = {}
        for layer_num in tqdm(range(len(projection_matrices))):
            domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
            domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
    
    # integrate the prior knowledge, domain knowledge and closed-form refinement
    cache_dict = {}
    for key in CFR_dict:
        cache_dict[key] = args.train_preserve_scale * (prior_preservation_cache_dict[key] \
                        + args.preserve_weight * domain_preservation_cache_dict[key]) \
                        + CFR_dict[key]
    
    # closed-form refinement
    projection_matrices, _, _ = get_ca_layers(unet, with_to_k=True)
    
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        closed_form_refinement(projection_matrices, lamb=args.lamb, preserve_scale=1, cache_dict=cache_dict)
    else:
        print(f"Starting closed-form refinement...")
        print(f"Using lambda: '{args.lamb}', preserve_scale: '{args.train_preserve_scale}'")
        closed_form_refinement(projection_matrices, contexts, valuess, lamb=args.lamb, 
                               preserve_scale=args.train_preserve_scale, cache_dict=cache_dict)
    
    del contexts, valuess, cache_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    # stage 2: multi-lora training
    for i in range(train_dataset._concept_num): # the number of concept/lora
        
        attn_controller = AttnController()
        if i != 0:
            unet.set_default_attn_processor()
        for name, m in unet.named_modules():
            if name.endswith('attn2') or name.endswith('attn1'):
                cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                m.set_processor(LoRAAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    rank=args.rank, 
                    attn_controller=attn_controller, 
                    module_name=name, 
                    preserve_prior=args.with_prior_preservation,
                ))

        ### set lora
        lora_attn_procs = {}
        for key, value in zip(unet.attn_processors.keys(), unet.attn_processors.values()):
            if key.endswith("attn2.processor"):
                lora_attn_procs[f'{key}.to_k_lora'] = value.to_k_lora
                lora_attn_procs[f'{key}.to_v_lora'] = value.to_v_lora
        
        lora_layers = AttnProcsLayers(lora_attn_procs)
            
        optimizer = optimizer_class(
            lora_layers.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("MACE")

        # Train
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running LoRA Finetuning *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size (per device) = {args.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Learning Rate = {args.learning_rate}")
        global_step = 0
        first_epoch = 0
        iteration = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
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
                resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

        if args.importance_sampling:
            print("""Using relation-focal importance sampling, which can make training more efficient
                  and is particularly beneficial in erasing mass concepts with overlapping terms.""")
            
            list_of_candidates = [
                x for x in range(noise_scheduler.config.num_train_timesteps)
            ]
            prob_dist = [
                importance_sampling_fn(x)
                for x in list_of_candidates
            ]
            prob_sum = 0
            # normalize the prob_list so that sum of prob is 1
            for j in prob_dist:
                prob_sum += j
            prob_dist = [x / prob_sum for x in prob_dist]
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
    
        debug_once = True
                
        if args.train_seperate:
            train_dataset.concept_number = i 
        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
                
            torch.cuda.empty_cache()
            gc.collect()
            
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step           
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # show
                    if debug_once:
                        print('==================================================================')
                        print(f'Concept {i}: {batch["instance_prompts"][0]}')
                        print('==================================================================')
                        debug_once = False
                        
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    if args.importance_sampling:
                        timesteps = np.random.choice(
                            list_of_candidates,
                            size=bsz,
                            replace=True,
                            p=prob_dist)
                        timesteps = torch.tensor(timesteps).cuda()
                    else:
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    if args.no_real_image:
                        noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)                
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    
                    # set concept_positions for this batch 
                    if args.use_gsam_mask:
                        GSAM_mask = batch['masks']
                    else:
                        GSAM_mask = None
                    
                    attn_controller.set_concept_positions(batch["concept_positions"], GSAM_mask, use_gsam_mask=args.use_gsam_mask)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    # Base attention loss
                    loss = attn_controller.loss()
                    attention_loss = loss.detach().clone()
                    
                    # Add regularization losses (from ContinualEnhancements)
                    l1sp_loss = None
                    l2sp_loss = None
                    inverse_ewc_loss = None
                    trajectory_loss = None
                    
                    if hasattr(args, 'l1sp_weight') and args.l1sp_weight > 0.0:
                        l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(unet, args.original_params)
                        loss = loss + l1sp_loss
                    
                    if hasattr(args, 'l2sp_weight') and args.l2sp_weight > 0.0:
                        l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(unet, args.original_params)
                        loss = loss + l2sp_loss
                    
                    if hasattr(args, 'inverse_ewc_lambda') and args.inverse_ewc_lambda > 0.0 and args.previous_aggregated_fisher is not None:
                        inverse_ewc_loss = args.inverse_ewc_lambda * calculate_inverse_ewc_loss(
                            unet, 
                            args.previous_aggregated_fisher, 
                            args.original_params, 
                            accelerator.device,
                            use_l2=getattr(args, 'inverse_ewc_use_l2', False)
                        )
                        loss = loss + inverse_ewc_loss
                    
                    if hasattr(args, 'trajectory_lambda') and args.trajectory_lambda > 0.0 and args.previous_aggregated_delta is not None:
                        trajectory_loss = args.trajectory_lambda * calculate_trajectory_loss(
                            unet,
                            args.previous_aggregated_delta,
                            args.original_params,
                            accelerator.device
                        )
                        loss = loss + trajectory_loss
                    
                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                        
                    accelerator.backward(loss)
                    
                    # Accumulate Fisher for this unlearning run (only used next unlearning run)
                    if hasattr(args, 'save_fisher_path') and args.save_fisher_path is not None:
                        args.current_fisher = accumulate_fisher(unet, args.current_fisher)
                    
                    # Apply gradient projection if enabled
                    if hasattr(args, 'with_gradient_projection') and args.with_gradient_projection and args.anchor_embeddings_matrix is not None:
                        apply_gradient_projection(
                            model=unet,
                            filtered_embedding_matrix=args.anchor_embeddings_matrix,
                            device=accelerator.device,
                            accelerator=accelerator
                        )
                    
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    attn_controller.zero_attn_probs()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    iteration += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                # Update logs with enhanced metrics
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                logs["attn_loss"] = attention_loss.item()
                if l1sp_loss is not None: logs["l1_loss"] = l1sp_loss.item()
                if l2sp_loss is not None: logs["l2_loss"] = l2sp_loss.item()
                if inverse_ewc_loss is not None: logs["ewc_loss"] = inverse_ewc_loss.item()
                if trajectory_loss is not None: logs["trajectory_loss"] = trajectory_loss.item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Simultaneous evaluation with early stopping
                if hasattr(args, 'eval_every') and args.eval_every is not None and iteration % args.eval_every == 0:
                    # Create sampling pipeline for evaluation
                    eval_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                    eval_text_encoder = copy.deepcopy(accelerator.unwrap_model(text_encoder))
                    if weight_dtype == torch.float32:
                        eval_unet.half()
                        eval_text_encoder.half()
                    
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=eval_unet,
                        text_encoder=eval_text_encoder,
                        tokenizer=tokenizer,
                        revision=args.revision,
                        torch_dtype=torch.float16,
                    ).to(accelerator.device)
                    pipeline.unet.eval()
                    pipeline.text_encoder.eval()
                    
                    # Sample and evaluate unlearned accuracy
                    save_path = os.path.join(args.output_dir, "delta.bin")
                    ua = sample_and_evaluate_ua(
                        pipeline, 
                        getattr(args, 'concept_type', 'object'),
                        iteration, 
                        save_path, 
                        getattr(args, 'prompt_list', None),
                        None, 
                        accelerator.device, 
                        getattr(args, 'classifier_dir', None)
                    )
                    print(f"Iteration '{iteration}', Unlearned Accuracy: '{ua}'")
                    
                    # Cleanup
                    del pipeline
                    torch.cuda.empty_cache()
                    
                    # Check for early stopping
                    args.best_ua, args.no_improvement_count, args.stop_training = check_early_stopping(
                        ua, args.best_ua, args.no_improvement_count, args.eval_every, args.patience
                    )
                    
                    if args.stop_training:
                        logger.info(f"Stopping training early at iteration {iteration} with best UA: {args.best_ua}")
                        break

                if global_step >= args.max_train_steps:
                    break
            
            # Break out of epoch loop if early stopping triggered
            if hasattr(args, 'stop_training') and args.stop_training:
                break

        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            ## save lora layers
            if args.train_seperate:
                concepts, _ = args.multi_concept[0][i]
            else:
                concepts = len(args.multi_concept[0])
                
            unet = accelerator.unwrap_model(unet).to(torch.float32)
            lora_path = f"{args.output_dir}/lora/{concepts}"
            os.makedirs(lora_path, exist_ok=True)
            print(f"Saving LoRA layers to {lora_path}")
            unet.save_attn_procs(lora_path)
            
            # Save Fisher Information if configured
            if hasattr(args, 'save_fisher_path') and args.save_fisher_path is not None and args.current_fisher:
                save_fisher_information(
                    args.current_fisher, 
                    args.save_fisher_path, 
                    iteration, 
                    args.previous_aggregated_fisher
                )
            
            # Save delta/trajectory if configured
            if hasattr(args, 'save_delta_path') and args.save_delta_path is not None:
                save_delta_to_path(
                    unet, 
                    args.original_params, 
                    args.save_delta_path, 
                    args.previous_aggregated_delta
                )
            
            # if isinstance(args, Namespace):
            #     with open(f"{args.output_dir}/my_args.json", "w") as f:
            #         json.dump(vars(args), f, indent=4)    

        accelerator.end_training()
        
        del lora_attn_procs, lora_layers, optimizer, lr_scheduler, attn_controller
        torch.cuda.empty_cache()

        if not args.train_seperate:
            break
    
    # save base initialized model 
    print(f"Saving closed-form refined model to {args.output_dir}")
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        revision=args.revision,
    )
    pipeline.save_pretrained(args.output_dir)