import torch


def get_training_params(unet, train_method='esd-x'):
    param_names = []
    param_values = []

    for name, module in unet.named_modules():
        if module.__class__.__name__ not in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            continue

        if train_method == 'esd-x' and 'attn2' in name:
            for n, p in module.named_parameters():
                param_names.append(f"{name}.{n}")
                param_values.append(p)

        if train_method == 'esd-u' and ('attn2' not in name):
            for n, p in module.named_parameters():
                param_names.append(f"{name}.{n}")
                param_values.append(p)

        if train_method == 'esd-all':
            for n, p in module.named_parameters():
                param_names.append(f"{name}.{n}")
                param_values.append(p)

        if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
            for n, p in module.named_parameters():
                param_names.append(f"{name}.{n}")
                param_values.append(p)

    if not param_values:
        raise ValueError(f"Unsupported train_method '{train_method}' for co-erasing.")

    return param_names, param_values


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def anneal_schedule(timestep, total_steps):
    x = torch.tensor(timestep / total_steps)
    return 1 / torch.exp(x)
