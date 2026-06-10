import argparse
import os


def parse_args(input_args=None):
    """Parse and normalize CLI arguments for ESD-SDXL training."""
    parser = argparse.ArgumentParser(
        prog="train_esd_sdxl.py",
        description="Train ESD on Stable Diffusion XL with CUIG regularizer support.",
    )

    # Concept erasure inputs.
    parser.add_argument("--concept", type=str, required=True, help="Concept to erase, list literal, or prompt file.")
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=["style", "object", "celeb", "celebrity"],
        help="Type of concept being erased.",
    )
    parser.add_argument("--erase_from", type=str, default=None, help="Optional prompt/concept to erase from.")
    parser.add_argument("--erase_from_prompts", type=str, default=None, help="Text file of erase-from prompts.")
    parser.add_argument("--negative_guidance", type=float, default=2.0, help="Negative guidance value for ESD.")

    # Inference/sampling settings used to build the ESD training signal.
    parser.add_argument("--num_inference_steps", type=int, default=50, help="SDXL denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG scale for latent sampling.")
    parser.add_argument("--height", type=int, default=1024, help="Generated latent height.")
    parser.add_argument("--width", type=int, default=1024, help="Generated latent width.")

    # Training settings.
    parser.add_argument(
        "--train_method",
        type=str,
        required=True,
        choices=["esd-x", "esd-u", "esd-all", "esd-x-strict"],
        help="ESD parameter subset to update.",
    )
    parser.add_argument("--iterations", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for CUIG-style outputs.")
    parser.add_argument("--save_path", type=str, default=None, help="Legacy file or directory save path.")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model path or Hugging Face ID.",
    )
    parser.add_argument("--unet_ckpt", "--unet_ckpt_path", dest="unet_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype used for SDXL components.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overwrite_existing_ckpt",
        "--overwrite",
        dest="overwrite_existing_ckpt",
        action="store_true",
        default=False,
        help="Overwrite an existing final checkpoint.",
    )

    # Simultaneous evaluation / early stopping.
    parser.add_argument("--eval_interval", "--eval_every", dest="eval_interval", type=int, default=None)
    parser.add_argument("--eval_start", type=int, default=0, help="First iteration eligible for evaluation.")
    parser.add_argument("--patience", type=int, default=2000, help="Early-stopping patience in iterations.")
    parser.add_argument("--stop_threshold", type=float, default=99.0, help="UA threshold for early stopping.")
    parser.add_argument("--eval_classifier_dir", "--classifier_dir", dest="eval_classifier_dir", type=str, default=None)
    parser.add_argument("--eval_prompt_dir", type=str, default=None, help="Prompt directory for celebrity/character eval.")

    # Weight regularizers.
    parser.add_argument("--l1sp_weight", type=float, default=0.0, help="Weight for L1-SP.")
    parser.add_argument("--l2sp_weight", type=float, default=0.0, help="Weight for L2-SP.")

    # SelFT.
    parser.add_argument("--with_selft", action="store_true", default=False, help="Apply SelFT gradient masking.")
    parser.add_argument("--selft_loss", type=str, default=None, help="SelFT importance loss: ESD or ConAbl.")
    parser.add_argument("--selft_topk", type=float, default=0.01, help="Top-k fraction of parameters for SelFT.")
    parser.add_argument("--selft_anchor", type=str, default="", help="Anchor concept(s) used by SelFT.")
    parser.add_argument("--selft_grad_dict_path", type=str, default=None, help="Path for SelFT grad cache.")
    parser.add_argument("--selft_mask_dict_path", type=str, default=None, help="Path for SelFT mask cache.")

    # Gradient projection.
    parser.add_argument("--with_gradient_projection", action="store_true", default=False)
    parser.add_argument(
        "--auxiliary_prompts_path",
        "--gradient_projection_prompts",
        dest="auxiliary_prompts_path",
        type=str,
        default=None,
        help="Auxiliary prompt file for gradient projection.",
    )
    parser.add_argument("--gradient_projection_num_prompts", type=int, default=400)
    parser.add_argument("--gradient_projection_no_dual_domain", action="store_true", default=False)
    parser.add_argument("--gradient_projection_llm_model_id", type=str, default="mistral")
    parser.add_argument("--previously_unlearned", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    _normalize_args(args)
    return args


def _normalize_args(args):
    """Convert legacy flags and aliases into the canonical runtime fields."""
    if args.concept_type == "celebrity":
        args.concept_type = "celeb"

    if args.selft_loss is not None:
        args.with_selft = True
        args.selft_loss = _normalize_selft_loss(args.selft_loss)
    elif args.with_selft:
        args.selft_loss = "ESD"

    output_dir_was_provided = args.output_dir is not None
    if args.output_dir is None:
        args.output_dir = _output_dir_from_save_path(args.save_path)

    if output_dir_was_provided:
        args.final_save_path = os.path.join(args.output_dir, "delta.bin")
    elif args.save_path is None:
        args.final_save_path = os.path.join(args.output_dir, "delta.bin")
    elif _looks_like_file_path(args.save_path):
        args.final_save_path = args.save_path
        args.output_dir = os.path.dirname(args.save_path) or "."
    else:
        args.output_dir = args.save_path
        args.final_save_path = os.path.join(args.output_dir, "delta.bin")

    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.final_save_path = os.path.abspath(os.path.expanduser(args.final_save_path))
    if args.unet_ckpt is not None:
        args.unet_ckpt = os.path.abspath(os.path.expanduser(args.unet_ckpt))


def _looks_like_file_path(path):
    if path is None:
        return False
    return os.path.splitext(path)[1] != ""


def _output_dir_from_save_path(save_path):
    if save_path is None:
        return "esd-models/sdxl"
    if _looks_like_file_path(save_path):
        return os.path.dirname(save_path) or "."
    return save_path


def _normalize_selft_loss(loss_name):
    normalized = loss_name.strip().lower()
    if normalized == "esd":
        return "ESD"
    if normalized in {"ca", "conabl"}:
        return "ConAbl"
    raise ValueError("Unsupported SelFT loss. Use 'ESD'/'esd' or 'ConAbl'/'ca'.")
