import os

import torch


def patch_sdxl_pipeline_call():
    """Install the ESD helper call that can stop denoising at an intermediate timestep."""
    from diffusers import StableDiffusionXLPipeline
    from src.sdxl_pipeline import esd_sdxl_call

    StableDiffusionXLPipeline.__call__ = esd_sdxl_call


def get_torch_dtype(torch_dtype_string):
    if torch_dtype_string == "float32":
        return torch.float32
    if torch_dtype_string == "bfloat16":
        return torch.bfloat16
    if torch_dtype_string == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {torch_dtype_string}")


def load_sdxl_models(args):
    """Load the frozen teacher UNet, trainable ESD UNet, and SDXL pipeline."""
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

    patch_sdxl_pipeline_call()
    torch_dtype = get_torch_dtype(args.torch_dtype)
    print(f"\nLoading SDXL from '{args.base_model_dir}' with dtype '{torch_dtype}' on '{args.device}'")

    base_unet = UNet2DConditionModel.from_pretrained(args.base_model_dir, subfolder="unet")
    esd_unet = UNet2DConditionModel.from_pretrained(args.base_model_dir, subfolder="unet")

    if args.unet_ckpt is not None:
        _load_unet_checkpoint(args.unet_ckpt, base_unet, esd_unet, args.base_model_dir)

    base_unet.to(args.device, dtype=torch_dtype)
    esd_unet.to(args.device, dtype=torch_dtype)
    base_unet.requires_grad_(False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model_dir,
        unet=base_unet,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    return pipe, base_unet, esd_unet, torch_dtype


def get_esd_trainable_parameters(esd_unet, train_method):
    """
    Select the trainable ESD parameter subset and freeze everything else.

    ESD naming:
    - esd-x: all cross-attention Linear/Conv parameters
    - esd-u: non-cross-attention Linear/Conv parameters
    - esd-all: all Linear/Conv parameters
    - esd-x-strict: cross-attention key/value parameters only
    """
    print(f"Training method: '{train_method}'")
    esd_unet.requires_grad_(False)

    trainable_names = []
    trainable_params = []
    module_types = {"Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"}

    for module_name, module in esd_unet.named_modules():
        if module.__class__.__name__ not in module_types:
            continue

        should_train = (
            (train_method == "esd-x" and "attn2" in module_name)
            or (train_method == "esd-u" and "attn2" not in module_name)
            or train_method == "esd-all"
            or (train_method == "esd-x-strict" and ("attn2.to_k" in module_name or "attn2.to_v" in module_name))
        )
        if not should_train:
            continue

        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param.requires_grad_(True)
            trainable_names.append(full_name)
            trainable_params.append(param)

    num_trainable = sum(param.numel() for param in trainable_params)
    print(f"Selected '{len(trainable_params)}' tensors with '{num_trainable:,}' trainable values.")
    if not trainable_params:
        raise ValueError(f"No trainable parameters selected for train_method '{train_method}'.")
    return trainable_names, trainable_params


def save_esd_checkpoint(trainable_names, trainable_params, save_path):
    """Save the selected ESD parameter tensors to disk."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    state_dict = {
        name: param.detach().cpu().clone()
        for name, param in zip(trainable_names, trainable_params)
    }
    if save_path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)
    print(f"Saved '{len(state_dict)}' ESD tensors to '{save_path}'")


def _load_unet_checkpoint(ckpt_path, base_unet, esd_unet, base_model_dir):
    """Load a previous CUIG/ESD UNet checkpoint into both teacher and student UNets."""
    if not os.path.exists(ckpt_path):
        print(f"UNet checkpoint not found at '{ckpt_path}'. Using default UNet from '{base_model_dir}'.")
        return

    print(f"Loading UNet checkpoint from '{ckpt_path}'")
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "unet" in state_dict:
        state_dict = state_dict["unet"]

    for model_name, unet in (("base", base_unet), ("esd", esd_unet)):
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        print(
            f"\tLoaded checkpoint into {model_name} UNet with "
            f"'{len(missing)}' missing and '{len(unexpected)}' unexpected keys."
        )
