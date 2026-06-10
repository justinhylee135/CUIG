import ast
import os
import random
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def check_for_existing_ckpt(args):
    """Create the output directory and check for overwrite request."""
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(args.final_save_path):
        if args.overwrite_existing_ckpt:
            print(f"Checkpoint '{args.final_save_path}' exists. Overwriting because requested.")
        else:
            print(f"Model already exists at '{args.final_save_path}'. Use '--overwrite_existing_ckpt' to overwrite.")
            sys.exit(0)


def set_seed(seed):
    """Seed Python and PyTorch RNGs."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeding RNGs with '{seed}'")


def prepare_concept_prompts(args):
    """Resolve target prompts and erase-from prompts used by ESD training."""
    args.target_concepts = _load_or_format_concepts(args.concept, args.concept_type)
    args.anchor_concepts = _broadcast_anchor(args.selft_anchor, len(args.target_concepts))
    args.erase_from_prompts_list = _load_erase_from_prompts(args.erase_from, args.erase_from_prompts)
    return args.target_concepts, args.erase_from_prompts_list


def configure_learning_rate(args):
    """Choose the ESD default learning rate when the user does not provide one."""
    if args.learning_rate is not None:
        print(f"\nUsing provided LR of '{args.learning_rate:.1e}' for '{args.concept_type}' unlearning")
        return
    if args.concept_type == "style":
        args.learning_rate = 1e-5
    elif args.concept_type == "object":
        args.learning_rate = 5e-6
    elif args.concept_type == "celeb":
        args.learning_rate = 2e-4
    else:
        raise ValueError(f"Unsupported concept_type '{args.concept_type}'")
    print(f"\nUsing default LR of '{args.learning_rate:.1e}' for '{args.concept_type}' unlearning")


def setup_regularizers(args, pipe, esd_unet):
    """Initialize CUIG regularizers and attach any gradient hooks."""
    print("Setting up Regularizers for Continual Unlearning...")
    args.original_params = {}
    grad_hooks = []

    if args.l1sp_weight > 0.0:
        print(f"\tUsing L1SP regularizer with weight '{args.l1sp_weight}'")
    if args.l2sp_weight > 0.0:
        print(f"\tUsing L2SP regularizer with weight '{args.l2sp_weight}'")

    if args.with_selft:
        from Regularizers.SelFT.selft import apply_selft_masks, get_selft_mask_dict

        if args.selft_mask_dict_path is None:
            args.selft_mask_dict_path = os.path.join(args.output_dir, "mask_dict.pt")
        if args.selft_grad_dict_path is None:
            args.selft_grad_dict_path = os.path.join(args.output_dir, "grad_dict.pt")

        print(f"\tUsing SelFT with loss '{args.selft_loss}' and top-k '{args.selft_topk}'")
        esd_unet.eval()
        selft_mask_dict = get_selft_mask_dict(
            esd_unet,
            pipe.text_encoder,
            pipe.tokenizer,
            args.selft_mask_dict_path,
            args.selft_grad_dict_path,
            args.target_concepts,
            args.anchor_concepts,
            args.selft_topk,
            args.selft_loss,
            args.device,
            diffusion_pipe=pipe,
        )
        grad_hooks = apply_selft_masks(esd_unet, selft_mask_dict)
        esd_unet.train()
        print(f"\tApplied SelFT masks with '{len(grad_hooks)}' hooks")

    if args.with_gradient_projection:
        _setup_gradient_projection(args, pipe)

    if args.eval_interval is not None:
        _validate_simultaneous_args(args)
        args.best_ua = 0.0
        args.no_improvement_count = 0
        args.early_stop_triggered = False
        print(
            f"\tUsing simultaneous evaluation every '{args.eval_interval}' iterations "
            f"after iteration '{args.eval_start}' with patience '{args.patience}'."
        )

    return grad_hooks


def apply_gradient_projection(unet, aux_orth_proj, device, accelerator=None):
    """Lazy wrapper around the shared projection regularizer."""
    from Regularizers.Projection.projection import apply_gradient_projection as _apply_gradient_projection

    return _apply_gradient_projection(
        unet=unet,
        aux_orth_proj=aux_orth_proj,
        device=device,
        accelerator=accelerator,
    )


def cleanup_hooks(grad_hooks):
    """Remove SelFT hooks registered on trainable parameters."""
    for hook in grad_hooks:
        hook.remove()
    if grad_hooks:
        print(f"Removed '{len(grad_hooks)}' SelFT gradient hooks")


def print_training_summary(args):
    """Print the compact training summary used before optimization starts."""
    print("***** Running ESD-SDXL training *****")
    print(f"  Base model = {args.base_model_dir}")
    print(f"  Target prompts = {args.target_concepts}")
    print(f"  Erase-from prompts = {args.erase_from_prompts_list or '[target prompts]'}")
    print(f"  Train method = {args.train_method}")
    print(f"  Iterations = {args.iterations}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Negative guidance = {args.negative_guidance}")
    print(f"  Guidance scale = {args.guidance_scale}")
    print(f"  Final checkpoint = {args.final_save_path}")


def truncate_for_log(text, max_len=48):
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _load_or_format_concepts(concept, concept_type):
    concept_path = os.path.abspath(os.path.expanduser(concept))
    if os.path.isfile(concept_path):
        prompts = _read_prompt_file(concept_path)
        print(f"Loaded '{len(prompts)}' target prompt(s) from '{concept_path}'")
        return prompts
    return process_input_concepts(concept, concept_type)


def _load_erase_from_prompts(erase_from, erase_from_prompts_path):
    if erase_from_prompts_path is not None:
        prompt_path = os.path.abspath(os.path.expanduser(erase_from_prompts_path))
        prompts = _read_prompt_file(prompt_path)
        print(f"Loaded '{len(prompts)}' erase-from prompt(s) from '{prompt_path}'")
        if erase_from is not None:
            print("Warning: both '--erase_from' and '--erase_from_prompts' were provided; using the prompt file.")
        return prompts
    if erase_from is not None:
        return [erase_from]
    return []


def _read_prompt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")
    return prompts


def _broadcast_anchor(anchor, num_targets):
    if isinstance(anchor, list):
        anchors = anchor
    elif isinstance(anchor, str) and "[" in anchor:
        anchors = ast.literal_eval(anchor)
    else:
        anchors = [anchor]
    if len(anchors) == 1:
        anchors = anchors * num_targets
    if len(anchors) != num_targets:
        raise ValueError(f"SelFT anchor count '{len(anchors)}' does not match target count '{num_targets}'")
    return anchors


def _setup_gradient_projection(args, pipe):
    from Regularizers.Projection.projection import build_orthogonal_projector, generate_auxiliary_prompts
    from Regularizers.Projection.src.auxiliary_embeddings import get_auxiliary_embeddings_from_diffusion_pipeline

    print("\tUsing Regularizer: Gradient Projection")
    auxiliary_prompts = []
    prompts_path = args.auxiliary_prompts_path

    if prompts_path is None:
        print("\t\tNo auxiliary prompt path provided; disabling gradient projection.")
        args.with_gradient_projection = False
        args.aux_orth_proj = None
        return

    prompts_path = os.path.abspath(os.path.expanduser(prompts_path))
    args.auxiliary_prompts_path = prompts_path
    if os.path.isfile(prompts_path):
        auxiliary_prompts = _read_prompt_file(prompts_path)
        print(f"\t\tLoaded '{len(auxiliary_prompts)}' auxiliary prompts from '{prompts_path}'")
    else:
        print(f"\t\tGenerating auxiliary prompts and saving to '{prompts_path}'")
        auxiliary_prompts = generate_auxiliary_prompts(
            auxiliary_prompts_path=prompts_path,
            num_prompts=args.gradient_projection_num_prompts,
            concept_type=args.concept_type,
            previously_unlearned=args.previously_unlearned,
            target_concepts=args.target_concepts.copy(),
            dual_domain=not args.gradient_projection_no_dual_domain,
            llm_model_id=args.gradient_projection_llm_model_id,
        )

    auxiliary_embeddings = get_auxiliary_embeddings_from_diffusion_pipeline(
        auxiliary_prompts,
        pipe,
        args.device,
    )
    args.aux_orth_proj = build_orthogonal_projector(auxiliary_embeddings, args.device)
    print(f"\t\tGradient projection matrix shape: '{args.aux_orth_proj.shape}'")


def _validate_simultaneous_args(args):
    if args.concept_type != "celeb" and args.eval_classifier_dir is None:
        raise ValueError("Set '--eval_classifier_dir' for simultaneous style/object evaluation.")
    if args.concept_type == "celeb" and args.eval_prompt_dir is None:
        raise ValueError("Set '--eval_prompt_dir' for simultaneous celebrity sampling.")


def process_input_concepts(concept, concept_type):
    """Format concept names into the prompts expected by ESD."""
    concepts = ast.literal_eval(concept) if "[" in concept else [concept]
    prompt_templates = {
        "style": lambda value: f"{value.replace('_', ' ')} Style",
        "object": lambda value: f"An image of {value}",
        "celeb": lambda value: value.replace("_", " "),
    }
    prompt_fn = prompt_templates.get(concept_type)
    if prompt_fn is None:
        raise ValueError(f"Unknown concept_type: {concept_type}")
    return [prompt_fn(value) for value in concepts]
