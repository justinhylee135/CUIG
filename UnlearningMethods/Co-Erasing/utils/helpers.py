import os
import torch


def save_model(model, save_path, idx: int = -1):
    state_dict = model.state_dict().copy()
    keys_to_remove = ['to_k_ip.weight', 'to_v_ip.adapter']
    for key in list(state_dict.keys()):
        if any(sub in key for sub in keys_to_remove):
            del state_dict[key]

    if os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        filename = f"unet_{idx}.pth" if idx != -1 else "unet.pth"
        target_path = os.path.join(save_path, filename)
    elif os.path.splitext(save_path)[1]:
        target_dir = os.path.dirname(save_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        target_path = save_path
    else:
        os.makedirs(save_path, exist_ok=True)
        filename = f"unet_{idx}.pth" if idx != -1 else "unet.pth"
        target_path = os.path.join(save_path, filename)

    print(f"Saving model at iteration {idx + 1} to '{target_path}'")
    torch.save(state_dict, target_path)


def to_same_device(tensors, device, dtype=None):
    return [
        t.to(device=device, dtype=dtype if dtype is not None else t.dtype)
        for t in tensors
    ]
