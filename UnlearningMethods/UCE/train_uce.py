import argparse
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from safetensors.torch import save_file, load_file
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ContinualEnhancements.SelFT.selft_utils import get_selft_mask_dict
from ContinualEnhancements.Projection.gradient_projection import (
    build_projection_matrix,
    generate_gradient_projection_prompts,
    get_anchor_embeddings,
)


def UCE(
    pipe,
    edit_concepts,
    guide_concepts,
    preserve_concepts,
    erase_scale,
    preserve_scale,
    lamb,
    save_dir,
    exp_name,
    *,
    with_gradient_projection=False,
    projection_matrix=None,
    projection_max_iters=1,
    projection_convergence_tol=None,
    l1sp_weight=0.0,
    l2sp_weight=0.0,
    selft_masks=None,
):
    start_time = time.time()
    # Prepare the cross attention weights required to do UCE
    uce_modules = []
    uce_module_names = []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)

    original_weight_map = {
        f"{name}.weight": module.weight.detach().clone().to(device=device, dtype=torch_dtype)
        for name, module in zip(uce_module_names, original_modules)
    }

    # collect text embeddings for erase concept and retain concepts
    uce_erase_embeds = {}
    for e in edit_concepts + guide_concepts + preserve_concepts:
        if e in uce_erase_embeds:
            continue
        t_emb = pipe.encode_prompt(prompt=e,
                                   device=device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)

        last_token_idx = (pipe.tokenizer(e,
                                          padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt",
                                         )['attention_mask']).sum()-2


        uce_erase_embeds[e] = t_emb[0][:,last_token_idx,:]

    # collect cross attention outputs for guide concepts and retain concepts (this is for original model weights)
    uce_guide_outputs = {}
    for g in guide_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue

        t_emb = uce_erase_embeds[g]

        for module in original_modules:
            uce_guide_outputs[g] = uce_guide_outputs.get(g, []) + [module(t_emb).detach()]

    progress_desc = "Editing"
    ###### UCE Algorithm (variables are named according to the paper: https://arxiv.org/abs/2308.14761)
    for module_idx, module in enumerate(tqdm(original_modules, desc=progress_desc)):
        # get original weight of the model
        w_old = module.weight.detach().clone().to(device=device, dtype=torch_dtype)

        base_mat1 = torch.zeros_like(w_old)
        base_mat2 = torch.zeros(
            (w_old.shape[1], w_old.shape[1]),
            device=w_old.device,
            dtype=w_old.dtype,
        )

        # Erase Concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = uce_erase_embeds[erase_concept].to(device=w_old.device, dtype=w_old.dtype).T
            v_i_star = uce_guide_outputs[guide_concept][module_idx].to(device=w_old.device, dtype=w_old.dtype).T

            base_mat1 += erase_scale * (v_i_star @ c_i.T)
            base_mat2 += erase_scale * (c_i @ c_i.T)

        # Retain Concepts
        for preserve_concept in preserve_concepts:
            c_i = uce_erase_embeds[preserve_concept].to(device=w_old.device, dtype=w_old.dtype).T
            v_i_star = uce_guide_outputs[preserve_concept][module_idx].to(device=w_old.device, dtype=w_old.dtype).T

            base_mat1 += preserve_scale * (v_i_star @ c_i.T)
            base_mat2 += preserve_scale * (c_i @ c_i.T)

        identity = torch.eye(w_old.shape[1], device=w_old.device, dtype=w_old.dtype)
        lambda_identity = lamb * identity

        layer_name = uce_module_names[module_idx]
        layer_key = f"{layer_name}.weight"
        layer_orig_weight = original_weight_map.get(layer_key, w_old)

        proj = None
        # Load orthogonal projector
        if with_gradient_projection and projection_matrix is not None:
            if isinstance(projection_matrix, dict):
                proj = (
                    projection_matrix.get(layer_key)
                    or projection_matrix.get(layer_name)
                )
            else:
                proj = projection_matrix
            if proj is not None:
                proj = proj.to(device=w_old.device, dtype=w_old.dtype)

        mask_tensor = None
        # Load SelFT mask
        if isinstance(selft_masks, dict):
            mask_tensor = (
                selft_masks.get(layer_key)
                or selft_masks.get(layer_name)
            )
            if mask_tensor is not None:
                mask_tensor = mask_tensor.to(device=w_old.device)
                if mask_tensor.dtype != w_old.dtype:
                    mask_tensor = mask_tensor.to(dtype=w_old.dtype)

        max_iters = max(int(projection_max_iters), 1)
        needs_iterative = (
            (proj is not None)
            or (mask_tensor is not None)
            or (l1sp_weight > 0.0)
            or (l2sp_weight > 0.0)
            or ((proj is not None) and (projection_convergence_tol is not None))
            or (max_iters > 1)
        )
        effective_iters = max_iters if needs_iterative else 1
        current_weight = w_old.clone()

        def apply_l1_regularizer(weight_tensor):
            if l1sp_weight <= 0.0:
                return weight_tensor
            delta_local = weight_tensor - layer_orig_weight
            shrink = torch.clamp(delta_local.abs() - l1sp_weight, min=0.0)
            shrink = shrink * delta_local.sign()
            return layer_orig_weight + shrink

        def apply_selft_mask(weight_tensor):
            if mask_tensor is None:
                return weight_tensor
            return mask_tensor * weight_tensor + (1.0 - mask_tensor) * layer_orig_weight

        progress_bar = tqdm(range(effective_iters), desc=f"Refining Layer {module_idx + 1}/{len(original_modules)}")
        for iter_idx in progress_bar:
            total_for_mat1 = base_mat1 + lamb * current_weight
            total_for_mat2 = base_mat2 + lambda_identity

            if l2sp_weight > 0.0:
                total_for_mat1 = total_for_mat1 + l2sp_weight * layer_orig_weight
                total_for_mat2 = total_for_mat2 + l2sp_weight * identity

            try:
                solved_weight_t = torch.linalg.solve(total_for_mat2.T, total_for_mat1.T)
                candidate_weight = solved_weight_t.T
            except RuntimeError:
                candidate_weight = total_for_mat1 @ torch.linalg.pinv(total_for_mat2)

            weight_without_projection = candidate_weight
            weight_with_projection = candidate_weight

            if proj is not None:
                base_weight = layer_orig_weight
                delta = weight_with_projection - base_weight
                projected_delta = delta @ proj
                weight_with_projection = base_weight + projected_delta

            weight_without_projection = apply_l1_regularizer(weight_without_projection)
            weight_with_projection = apply_l1_regularizer(weight_with_projection)

            weight_without_projection = apply_selft_mask(weight_without_projection)
            weight_with_projection = apply_selft_mask(weight_with_projection)

            convergence_gap = None
            if proj is not None and projection_convergence_tol is not None:
                diffs = []
                with torch.no_grad():
                    for concept in edit_concepts:
                        embed = uce_erase_embeds[concept].to(device=w_old.device, dtype=w_old.dtype)
                        output_proj = F.linear(embed, weight_with_projection, module.bias)
                        output_no_proj = F.linear(embed, weight_without_projection, module.bias)
                        diff = torch.norm(output_proj - output_no_proj, p=2)
                        diffs.append(diff.item())
                if diffs:
                    convergence_gap = max(diffs)
                    progress_bar.set_postfix({"proj_gap": f"{convergence_gap:.3e}"})
                else:
                    convergence_gap = 0.0

            current_weight = weight_with_projection.detach()

            if convergence_gap is not None and convergence_gap <= projection_convergence_tol:
                break

        uce_modules[module_idx].weight = torch.nn.Parameter(current_weight.to(dtype=torch_dtype))

    # save the weights
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name+'.weight'] = parameter.weight
    print(f"Saving UCE model weights to {os.path.join(save_dir, exp_name+'.safetensors')}")
    save_file(uce_state_dict, os.path.join(save_dir, exp_name+'.safetensors'))

    end_time = time.time()
    print(f'\n\nErased concepts using UCE\nModel edited in {end_time-start_time} seconds\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUCE',
                    description = 'UCE for erasing concepts in Stable Diffusion')
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object'], type=str, required=True)

    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="CompVis/stable-diffusion-v1-4",)
    parser.add_argument('--unet_ckpt_path', help='path to unet checkpoint if using a finetuned model', type=str, default=None)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')

    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='lambda regularization term for UCE', type=float, required=False, default=0.5)

    parser.add_argument('--expand_prompts', help='do you wish to expand your prompts?', choices=['true', 'false'], type=str, required=False, default='false')
    parser.add_argument('--with_gradient_projection', action='store_true', default=False, help='Apply gradient projection during UCE refinement')
    parser.add_argument('--gradient_projection_prompts', type=str, default=None, help='Path to anchor prompts txt for gradient projection')
    parser.add_argument('--gradient_projection_num_prompts', type=int, default=400, help='Number of prompts to generate for gradient projection')
    parser.add_argument('--previously_unlearned', type=str, default=None, help='Previously unlearned concepts to exclude from projection prompts')
    parser.add_argument('--projection_max_iters', type=int, default=1, help='Maximum iterations for projection-constrained refinement')
    parser.add_argument('--projection_convergence_tol', type=float, default=1e-4, help='L2 tolerance between projected and unprojected KV outputs for convergence')
    parser.add_argument('--l1sp_weight', type=float, default=0.0, help='Weight for L1-SP regularizer (shrinkage towards original weights)')
    parser.add_argument('--l2sp_weight', type=float, default=0.0, help='Weight for L2-SP regularizer (quadratic pull towards original weights)')
    parser.add_argument('--selft_loss', type=str, default=None, choices=['esd', 'ca'], help='Type of SelFT importance loss to use')
    parser.add_argument('--selft_topk', type=float, default=0.01, help='Top-k percentage of parameters by importance')
    parser.add_argument('--selft_anchor', type=str, default="", help='Anchor concept for SelFT CA loss')
    parser.add_argument('--selft_grad_dict_path', type=str, default=None, help='Path to save/load SelFT gradient dictionary')
    parser.add_argument('--selft_mask_dict_path', type=str, default=None, help='Path to save/load SelFT mask dictionary')

    parser.add_argument('--save_dir', help='where to save your uce model weights', type=str, default='../uce_models')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)

    args = parser.parse_args()

    device = args.device
    torch_dtype = torch.float32
    model_id = args.model_id

    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb

    concept_type = args.concept_type
    expand_prompts = args.expand_prompts

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'uce_test'

    # erase concepts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    # guide concepts
    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        guide_concepts = ''
        if concept_type == 'art':
            guide_concepts = 'art'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts*len(edit_concepts)
    if len(guide_concepts) != len(edit_concepts):
        raise Exception('Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are seperated by ; and are of equal sizes')

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]

    if expand_prompts == 'true':
        preserve_concepts = []
        edit_concepts = [concept for concept in edit_concepts if concept]
        guide_concepts = [concept for concept in guide_concepts if concept]
        if concept_type == 'art':
            expanded_edit_concepts = []
            expanded_guide_concepts = []
            for concept, guide_concept in zip(edit_concepts, guide_concepts):
                expanded_edit_concepts.extend([f'{concept}',
                                               f'painting by {concept}',
                                               f'art by {concept}',
                                               f'artwork by {concept}',
                                               f'picture by {concept}',
                                               f'style of {concept}'
                                              ]
                                             )
                expanded_guide_concepts.extend([f'{guide_concept}',
                                               f'painting by {guide_concept}',
                                               f'art by {guide_concept}',
                                               f'artwork by {guide_concept}',
                                               f'picture by {guide_concept}',
                                               f'style of {guide_concept}'
                                              ]
                                             )
            edit_concepts = expanded_edit_concepts
            guide_concepts = expanded_guide_concepts
        else:
            expanded_edit_concepts = []
            expanded_guide_concepts = []
            for concept, guide_concept in zip(edit_concepts, guide_concepts):
                expanded_edit_concepts.extend([f'{concept}',
                                               f'image of {concept}',
                                               f'photo of {concept}',
                                               f'portrait of {concept}',
                                               f'picture of {concept}',
                                               f'painting of {concept}'
                                              ]
                                             )
                expanded_guide_concepts.extend([f'{guide_concept}',
                                               f'image of {guide_concept}',
                                               f'photo of {guide_concept}',
                                               f'portrait of {guide_concept}',
                                               f'picture of {guide_concept}',
                                               f'painting of {guide_concept}'
                                              ]
                                             )
            edit_concepts = expanded_edit_concepts
            guide_concepts = expanded_guide_concepts

    print(f"Using erase scale of '{erase_scale}', lamb of '{lamb}', and preserve scale of '{preserve_scale}'")
    print(f"\n\nErasing: {edit_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        use_safetensors=True
    ).to(device)

    if args.unet_ckpt_path is not None:
        if "safetensors" in args.unet_ckpt_path:
            unet_state_dict = load_file(args.unet_ckpt_path)
        else:
            unet_state_dict = torch.load(args.unet_ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = pipe.unet.load_state_dict(unet_state_dict, strict=False)
        print(f"Loaded UNet weights from '{args.unet_ckpt_path}' with missing keys: '{missing_keys}' and unexpected keys: '{unexpected_keys}'")

    projection_matrix = None
    if args.with_gradient_projection:
        anchor_prompts = []
        if args.gradient_projection_prompts:
            if os.path.isfile(args.gradient_projection_prompts):
                with open(args.gradient_projection_prompts, 'r') as f:
                    prompts = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded '{len(prompts)}' anchor prompts from '{args.gradient_projection_prompts}'")
                anchor_prompts.extend(prompts)
            else:
                print(f"Generating gradient projection prompts and saving to file: '{args.gradient_projection_prompts}'")
                anchor_prompts = generate_gradient_projection_prompts(
                    file_path=args.gradient_projection_prompts,
                    num_prompts=args.gradient_projection_num_prompts,
                    concept_type=args.concept_type,
                    previously_unlearned=args.previously_unlearned,
                    target_concept_list=edit_concepts.copy(),
                )
        else:
            raise ValueError("Gradient projection requested but no gradient_projection_prompts path was provided.")

        if len(anchor_prompts) == 0:
            raise ValueError("No anchor prompts available to build gradient projection matrix.")

        print(f"Total anchor prompts collected: '{len(anchor_prompts)}'")
        anchor_embeddings_matrix = get_anchor_embeddings(
            anchor_prompts,
            pipe.text_encoder,
            pipe.tokenizer,
            device,
        )
        projection_matrix = build_projection_matrix(anchor_embeddings_matrix, device)
        if projection_matrix is None:
            raise ValueError("Failed to build gradient projection matrix from anchor prompts.")
        projection_matrix = projection_matrix.to(dtype=torch_dtype)
        print(f"Built gradient projection matrix of shape {projection_matrix.shape}")

    selft_masks = None
    if args.selft_loss is not None:
        print(f"Using SelFT with loss type '{args.selft_loss}' and top-k '{args.selft_topk}'")
        pipe.unet.eval()
        prompt_list = [p for p in (edit_concepts + guide_concepts + preserve_concepts) if p]
        if not prompt_list:
            raise ValueError("SelFT requested but no prompts available to compute importance masks.")
        selft_masks = get_selft_mask_dict(
            pipe.unet,
            pipe.text_encoder,
            pipe.tokenizer,
            args.selft_mask_dict_path,
            args.selft_grad_dict_path,
            prompt_list,
            args.selft_anchor,
            args.selft_topk,
            args.selft_loss,
            device,
        )
        pipe.unet.train()

    if args.l1sp_weight > 0.0:
        print(f"Applying L1-SP regularizer with weight {args.l1sp_weight}")
    if args.l2sp_weight > 0.0:
        print(f"Applying L2-SP regularizer with weight {args.l2sp_weight}")

    UCE(
        pipe,
        edit_concepts,
        guide_concepts,
        preserve_concepts,
        erase_scale,
        preserve_scale,
        lamb,
        save_dir,
        exp_name,
        with_gradient_projection=args.with_gradient_projection,
        projection_matrix=projection_matrix,
        projection_max_iters=args.projection_max_iters,
        projection_convergence_tol=args.projection_convergence_tol,
        l1sp_weight=args.l1sp_weight,
        l2sp_weight=args.l2sp_weight,
        selft_masks=selft_masks,
    )
