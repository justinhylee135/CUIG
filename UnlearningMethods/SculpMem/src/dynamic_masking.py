"""Dynamic attention masking utilities for SculpMem.

These hooks keep only a moving subset of trainable attention weights active.
The active set is initialized from accumulated gradients after warmup, then
periodically refreshed by swapping low-importance active weights with
high-importance inactive weights.
"""

import torch
import torch.nn as nn


def _weight_mask_hook(module, param_name, ratio):
    """Accumulate gradients and apply the module's current binary mask."""
    def hook(grad):
        if ratio == 0:
            return torch.zeros_like(grad)

        if hasattr(module, "accumulated_grad_weight"):
            module.accumulated_grad_weight += grad.detach()
        else:
            module.accumulated_grad_weight = grad.detach().clone()

        if hasattr(module, "mask"):
            return grad * module.mask.to(device=grad.device, dtype=grad.dtype)
        return grad

    return hook


def _register_linear_hook(linear_module, param_name, ratio):
    """Register a masking hook once for a trainable linear projection."""
    if getattr(linear_module, "sculpmem_mask_hook_registered", False):
        return False

    linear_module.weight.register_hook(_weight_mask_hook(linear_module, param_name, ratio))
    linear_module.sculpmem_mask_hook_registered = True
    return True


def register_attention_hooks(unet, update_percent_dict, cross_attention_only=True):
    """Register dynamic masking hooks on trainable attention projections."""
    hook_counts = {"q": 0, "k": 0, "v": 0, "to_out": 0}
    total_trainable = 0
    selected_trainable = 0

    for name, module in unet.named_modules():
        if not all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v", "to_out")):
            continue
        if cross_attention_only and "attn2" not in name:
            continue

        projections = (
            ("q", module.to_q, f"{name}.to_q.weight"),
            ("k", module.to_k, f"{name}.to_k.weight"),
            ("v", module.to_v, f"{name}.to_v.weight"),
            ("to_out", module.to_out[0], f"{name}.to_out.0.weight"),
        )

        for key, linear_module, param_name in projections:
            if not isinstance(linear_module, nn.Linear):
                continue
            if not linear_module.weight.requires_grad:
                continue

            ratio = update_percent_dict.get(key, 1.0)
            total_trainable += linear_module.weight.numel()
            selected_trainable += int(ratio * linear_module.weight.numel())

            if ratio < 1.0 and _register_linear_hook(linear_module, param_name, ratio):
                hook_counts[key] += 1

    coverage = 100.0 * selected_trainable / total_trainable if total_trainable else 0.0
    print(f"SculpMem dynamic mask hooks registered: {hook_counts}")
    print(f"SculpMem dynamic mask coverage: {selected_trainable}/{total_trainable} ({coverage:.2f}%)")


def update_masks_for_unet(unet, update_percent_dict, turnover_fraction, cross_attention_only=True):
    """Update masks from accumulated gradient magnitudes."""
    init_counts = {"q": 0, "k": 0, "v": 0, "to_out": 0}
    update_counts = {"q": 0, "k": 0, "v": 0, "to_out": 0}
    change_stats = {"q": [0, 0], "k": [0, 0], "v": [0, 0], "to_out": [0, 0]}

    def update_linear_module(linear_module, update_percent, key):
        if not hasattr(linear_module, "accumulated_grad_weight"):
            print("Skipping dynamic mask update; no accumulated gradient is available yet.")
            return

        importance = linear_module.accumulated_grad_weight.abs()
        total_elements = importance.numel()
        importance_flat = importance.reshape(-1)

        if not hasattr(linear_module, "mask"):
            k = max(1, int(update_percent * total_elements))
            threshold = importance_flat.kthvalue(total_elements - k + 1).values
            linear_module.mask = (importance >= threshold).float()
            init_counts[key] += 1
        else:
            current_mask = linear_module.mask.reshape(-1)
            active_indices = (current_mask == 1).nonzero(as_tuple=False).reshape(-1)
            inactive_indices = (current_mask == 0).nonzero(as_tuple=False).reshape(-1)

            if active_indices.numel() == 0:
                print("Skipping dynamic mask turnover; no active weights remain.")
            else:
                update_count = max(1, int(turnover_fraction * active_indices.numel()))

                active_importance = importance_flat[active_indices]
                _, smallest_active = torch.topk(active_importance, k=update_count, largest=False)
                drop_indices = active_indices[smallest_active]

                if inactive_indices.numel() > 0:
                    inactive_importance = importance_flat[inactive_indices]
                    add_count = min(update_count, inactive_indices.numel())
                    _, largest_inactive = torch.topk(inactive_importance, k=add_count, largest=True)
                    add_indices = inactive_indices[largest_inactive]
                else:
                    add_indices = torch.empty(0, dtype=torch.long, device=importance_flat.device)

                new_mask = current_mask.clone()
                new_mask[drop_indices] = 0.0
                new_mask[add_indices] = 1.0

                changed = (new_mask != current_mask).sum().item()
                total = current_mask.numel()
                linear_module.mask = new_mask.reshape_as(linear_module.mask)
                update_counts[key] += 1
                change_stats[key][0] += changed
                change_stats[key][1] += total

        linear_module.accumulated_grad_weight.zero_()

    for name, module in unet.named_modules():
        if cross_attention_only and "attn2" not in name:
            continue
        if not all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v", "to_out")):
            continue

        projections = (
            ("q", module.to_q),
            ("k", module.to_k),
            ("v", module.to_v),
            ("to_out", module.to_out[0]),
        )

        for key, linear_module in projections:
            if not isinstance(linear_module, nn.Linear):
                continue
            if not linear_module.weight.requires_grad:
                continue
            if update_percent_dict.get(key, 0.0) <= 0.0:
                continue

            update_linear_module(linear_module, update_percent_dict[key], key)

    pct_change = {
        key: f"{100.0 * changed / total:.2f}%" if total else "0.00%"
        for key, (changed, total) in change_stats.items()
    }
    print(f"SculpMem dynamic mask init counts: {init_counts}")
    print(f"SculpMem dynamic mask update counts: {update_counts}")
    print(f"SculpMem dynamic mask change percentages: {pct_change}")
