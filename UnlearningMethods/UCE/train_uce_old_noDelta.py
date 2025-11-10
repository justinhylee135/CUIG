"""Legacy implementation of the UCE editing routine for Stable Diffusion."""

from __future__ import annotations

import argparse
import ast
import operator
import os
import sys
from functools import reduce
from typing import Dict, List, Optional, Sequence

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from safetensors.torch import load_file

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ContinualEnhancements.Projection.gradient_projection import (
    build_projection_matrix,
    generate_gradient_projection_prompts,
    get_anchor_embeddings,
)


def _collect_cross_attention_layers(unet: torch.nn.Module) -> List[torch.nn.Module]:
    """Return all cross-attention blocks (attn2 modules) from the UNet."""
    layers: List[torch.nn.Module] = []
    for name, module in unet.named_children():
        if "up" in name or "down" in name:
            for block in module:
                if "Cross" not in block.__class__.__name__:
                    continue
                for attn in block.attentions:
                    for transformer in attn.transformer_blocks:
                        layers.append(transformer.attn2)
        if "mid" in name:
            for attn in module.attentions:
                for transformer in attn.transformer_blocks:
                    layers.append(transformer.attn2)
    return layers


def _find_module_name(named_modules: Dict[str, torch.nn.Module], target: torch.nn.Module) -> str:
    """Locate the registered module name for a given module instance."""
    for name, module in named_modules.items():
        if module is target:
            return name
    raise ValueError("Module name not found for target module.")


def _collect_projection_modules(
    unet: torch.nn.Module,
    ca_layers: Sequence[torch.nn.Module],
    *,
    with_to_k: bool,
) -> tuple[List[torch.nn.Module], List[str]]:
    """Gather projection modules (to_v and optionally to_k) and their names."""
    named_modules = dict(unet.named_modules())
    projection_modules: List[torch.nn.Module] = []
    module_names: List[str] = []

    for layer in ca_layers:
        to_v = layer.to_v
        projection_modules.append(to_v)
        module_names.append(_find_module_name(named_modules, to_v))

    if with_to_k:
        for layer in ca_layers:
            to_k = layer.to_k
            projection_modules.append(to_k)
            module_names.append(_find_module_name(named_modules, to_k))

    return projection_modules, module_names


def _literal_eval_if_needed(value):
    """Evaluate incoming string values if they encode literals."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _tokenize_pair(ldm_pipeline, text_a: str, text_b: str):
    """Tokenize two prompts and return trimmed terminal token embeddings."""
    text_input = ldm_pipeline.tokenizer(
        [text_a, text_b],
        padding="max_length",
        max_length=ldm_pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = ldm_pipeline.text_encoder(
        text_input.input_ids.to(ldm_pipeline.device)
    )[0]

    final_token_idx_a = text_input.attention_mask[0].sum().item() - 2
    final_token_idx_b = text_input.attention_mask[1].sum().item() - 2
    farthest = max(final_token_idx_a, final_token_idx_b)

    emb_a = text_embeddings[0]
    emb_b = text_embeddings[1]
    trimmed_a = emb_a[
        final_token_idx_a : len(emb_a) - max(0, farthest - final_token_idx_a)
    ]
    trimmed_b = emb_b[
        final_token_idx_b : len(emb_b) - max(0, farthest - final_token_idx_b)
    ]
    return trimmed_a, trimmed_b


def _compute_value_projections(
    projection_matrices: Sequence[torch.nn.Module],
    old_emb: torch.Tensor,
    new_emb: torch.Tensor,
    technique: str,
) -> List[torch.Tensor]:
    """Project embeddings through the KV matrices according to the chosen technique."""
    values: List[torch.Tensor] = []
    with torch.no_grad():
        for module in projection_matrices:
            if technique == "tensor":
                original = module(old_emb).detach()
                direction = original / original.norm()

                new_projected = module(new_emb).detach()
                projection = (direction * new_projected).sum()
                target = new_projected - projection * direction
                values.append(target.detach())
            else:
                values.append(module(new_emb).detach())
    return values


def _accumulate_matrices(
    context: torch.Tensor,
    values: Sequence[torch.Tensor],
    target_idx: int,
    scale: float,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    projection_matrix: Optional[torch.Tensor] = None,
) -> None:
    """Update the running covariance matrices used by the closed-form edit."""
    if projection_matrix is not None:
        context = context @ projection_matrix
    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
    context_vector_t = context.reshape(context.shape[0], 1, context.shape[1])
    value_vector = values[target_idx].reshape(
        values[target_idx].shape[0], values[target_idx].shape[1], 1
    )
    mat1 += scale * (value_vector @ context_vector_t).sum(dim=0)
    mat2 += scale * (context_vector @ context_vector_t).sum(dim=0)


def edit_model(
    ldm_stable,
    old_texts: Sequence[str],
    new_texts: Sequence[str],
    retain_texts: Sequence[str],
    add: bool = False,  # kept for backwards compatibility; unused
    layers_to_edit=None,
    lamb: float = 0.1,
    erase_scale: float = 0.1,
    preserve_scale: float = 0.1,
    with_to_k: bool = True,
    technique: str = "tensor",
    with_gradient_projection: bool = False,
    projection_matrix: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None,
):
    """Apply closed-form UCE edits directly on the UNet cross-attention projections."""
    ca_layers = _collect_cross_attention_layers(ldm_stable.unet)
    projection_modules, module_names = _collect_projection_modules(
        ldm_stable.unet, ca_layers, with_to_k=with_to_k
    )
    original_weight_map = {
        name: module.weight.detach().clone()
        for name, module in zip(module_names, projection_modules)
    }

    layers_to_edit = _literal_eval_if_needed(layers_to_edit)
    lamb = _literal_eval_if_needed(lamb)

    sanitized_old_texts = []
    sanitized_new_texts = []
    for old_text, new_text in zip(old_texts, new_texts):
        sanitized_old_texts.append(old_text)
        sanitized_new_texts.append(new_text or " ")

    retain_texts = list(retain_texts or [""])

    print(f"Editing prompts {sanitized_old_texts} -> {sanitized_new_texts}")
    for layer_idx in tqdm(
        range(len(projection_modules)), desc="Editing layers", leave=True
    ):
        if layers_to_edit is not None and layer_idx not in layers_to_edit:
            continue
        with torch.no_grad():
            module = projection_modules[layer_idx]
            weight_device = module.weight.device
            weight_dtype = module.weight.dtype

            layer_name = module_names[layer_idx]
            original_weight = original_weight_map[layer_name].to(
                device=weight_device, dtype=weight_dtype
            )

            identity = torch.eye(
                module.weight.shape[1],
                device=weight_device,
                dtype=weight_dtype,
            )
            mat1 = lamb * original_weight
            mat2 = lamb * identity

            projector = None
            if with_gradient_projection and projection_matrix is not None:
                proj = projection_matrix
                if isinstance(projection_matrix, dict):
                    proj = (
                        projection_matrix.get(f"{layer_name}.weight")
                        or projection_matrix.get(layer_name)
                    )
                if proj is not None:
                    proj = proj.to(device=weight_device, dtype=weight_dtype)
                    projector = proj

            for old_text, new_text in zip(sanitized_old_texts, sanitized_new_texts):
                old_emb, new_emb = _tokenize_pair(ldm_stable, old_text, new_text)
                values = _compute_value_projections(
                    projection_modules, old_emb, new_emb, technique
                )
                _accumulate_matrices(
                    old_emb.detach(),
                    values,
                    layer_idx,
                    erase_scale,
                    mat1,
                    mat2,
                    projector,
                )

            for retain_text in retain_texts:
                retain_old, retain_new = _tokenize_pair(ldm_stable, retain_text, retain_text)
                values = _compute_value_projections(
                    projection_modules, retain_old, retain_new, "replace"
                )
                _accumulate_matrices(
                    retain_old.detach(),
                    values,
                    layer_idx,
                    preserve_scale,
                    mat1,
                    mat2,
                    projector,
                )
            try:
                solved_weight_t = torch.linalg.solve(mat2.T, mat1.T)
                updated_weight = solved_weight_t.T
            except RuntimeError:
                updated_weight = mat1 @ torch.linalg.pinv(mat2)

            # if projector is not None:
            #     delta = updated_weight - original_weight
            #     updated_weight = original_weight + delta @ projector

            module.weight = torch.nn.Parameter(updated_weight)

    print(
        f'Current model status: Edited "{sanitized_old_texts}" into '
        f'"{sanitized_new_texts}" and retained "{retain_texts}"'
    )
    return ldm_stable


def parse_args():
    """Build argument parser for the legacy training script."""
    parser = argparse.ArgumentParser(
        prog="TrainUCE", description="Finetuning stable diffusion to debias concepts"
    )
    parser.add_argument(
        "--edit_concepts",
        help="prompt corresponding to concept to erase; comma separated",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--guide_concepts",
        help="concepts to guide the erased concepts towards; comma separated",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--preserve_concepts",
        help="concepts to preserve; comma separated",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--technique",
        help="technique to erase (either replace or tensor)",
        type=str,
        required=False,
        default="replace",
    )
    parser.add_argument(
        "--preserve_scale",
        help="scale to preserve concepts",
        type=float,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--preserve_number",
        help="number of preserve concepts",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--erase_scale",
        help="scale to erase concepts",
        type=float,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--concept_type",
        help="type of concept being erased",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--expand_prompts",
        help="expand prompts with common templates",
        choices=["true", "false"],
        type=str,
        required=False,
        default="false",
    )
    parser.add_argument(
        "--model_id",
        help="base version for stable diffusion",
        type=str,
        required=False,
        default="1.4",
    )
    parser.add_argument(
        "--device",
        help="cuda device index",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        help="path to UNet checkpoint if using a finetuned model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--with_gradient_projection",
        action="store_true",
        default=False,
        help="Apply gradient projection during editing",
    )
    parser.add_argument(
        "--gradient_projection_prompts",
        type=str,
        default=None,
        help="Path to anchor prompts txt for gradient projection",
    )
    parser.add_argument(
        "--gradient_projection_num_prompts",
        type=int,
        default=200,
        help="Number of prompts to generate for gradient projection",
    )
    parser.add_argument(
        "--previously_unlearned",
        type=str,
        default=None,
        help="Previously unlearned concepts to exclude from projection prompts",
    )
    parser.add_argument(
        "--save_path",
        help="Full path where the edited UNet weights should be stored",
        type=str,
        default=None,
    )
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    technique = args.technique
    device = f"cuda:{args.device}"
    with_gradient_projection = args.with_gradient_projection

    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    guided_concepts = args.guide_concepts
    preserve_concepts = args.preserve_concepts

    concepts = [concept.strip() for concept in args.edit_concepts.split(",")]
    concept_type = args.concept_type

    print_text = "_".join(concept.lower() for concept in concepts)

    old_texts = [concept for concept in concepts]

    if guided_concepts is None:
        new_texts = [" " for _ in old_texts]
        print_text += "-towards_uncond"
    else:
        guided_concepts = [concept.strip() for concept in guided_concepts.split(",")]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text += f"-towards_{guided_concepts[0]}"
        else:
            new_texts = reduce(operator.concat, [[concept] for concept in guided_concepts])
            print_text += "-towards"
            for target in new_texts:
                if target not in print_text:
                    print_text += f"-{target}"

    assert len(new_texts) == len(old_texts)

    if preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in preserve_concepts.split(",")]

    retain_texts = [""] + preserve_concepts
    print_text += "-preserve_true" if len(retain_texts) > 1 else "-preserve_false"

    if preserve_scale is None:
        preserve_scale = max(0.1, 1 / len(retain_texts))

    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_id).to(device)
    if args.unet_ckpt_path is not None:
        if not os.path.exists(args.unet_ckpt_path):
            print(f"UNet checkpoint not found at '{args.unet_ckpt_path}'. Using default UNet from pipeline '{args.model_id}'...")
        else: 
            if "safetensors" in args.unet_ckpt_path:
                unet_state_dict = load_file(args.unet_ckpt_path)
            else:
                unet_state_dict = torch.load(args.unet_ckpt_path, map_location="cpu")
            missing_keys, unexpected_keys = ldm_stable.unet.load_state_dict(
                unet_state_dict, strict=False
            )
            print(
                f"Loaded UNet weights from '{args.unet_ckpt_path}' with missing keys: "
                f"'{missing_keys}' and unexpected keys: '{unexpected_keys}'"
            )

    projection_matrix: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None

    if with_gradient_projection:
        anchor_prompts: List[str] = []
        if args.gradient_projection_prompts:
            if os.path.isfile(args.gradient_projection_prompts):
                with open(args.gradient_projection_prompts, "r", encoding="utf-8") as file:
                    prompts = [line.strip() for line in file.readlines() if line.strip()]
                print(
                    f"Loaded '{len(prompts)}' anchor prompts from "
                    f"'{args.gradient_projection_prompts}'"
                )
                anchor_prompts.extend(prompts)
            else:
                print(
                    f"Generating gradient projection prompts and saving to file: "
                    f"'{args.gradient_projection_prompts}'"
                )
                anchor_prompts = generate_gradient_projection_prompts(
                    file_path=args.gradient_projection_prompts,
                    num_prompts=args.gradient_projection_num_prompts,
                    concept_type=concept_type,
                    previously_unlearned=args.previously_unlearned,
                    target_concept_list=old_texts.copy(),
                )
        else:
            raise ValueError(
                "Gradient projection requested but no gradient_projection_prompts path was provided."
            )

        if len(anchor_prompts) == 0:
            raise ValueError("No anchor prompts available to build gradient projection matrix.")

        print(f"Total anchor prompts collected: '{len(anchor_prompts)}'")
        anchor_embeddings_matrix = get_anchor_embeddings(
            anchor_prompts,
            ldm_stable.text_encoder,
            ldm_stable.tokenizer,
            device,
        )
        projection_matrix = build_projection_matrix(anchor_embeddings_matrix, device)
        if projection_matrix is None:
            raise ValueError("Failed to build gradient projection matrix from anchor prompts.")
        weight_dtype = next(ldm_stable.unet.parameters()).dtype
        projection_matrix = projection_matrix.to(dtype=weight_dtype)

    print_text += f"-sd_{args.model_id.replace('.', '_')}"
    print_text += f"-method_{technique}"
    print(print_text.lower())

    ldm_stable = edit_model(
        ldm_stable=ldm_stable,
        old_texts=old_texts,
        new_texts=new_texts,
        add=False,
        retain_texts=retain_texts,
        lamb=0.5,
        erase_scale=erase_scale,
        preserve_scale=preserve_scale,
        technique=technique,
        with_gradient_projection=with_gradient_projection,
        projection_matrix=projection_matrix,
    )

    if args.save_path:
        dir_path = os.path.dirname(args.save_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        torch.save(ldm_stable.unet.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
