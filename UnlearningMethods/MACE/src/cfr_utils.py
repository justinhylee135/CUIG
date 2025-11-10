import copy
import torch
import numpy as np
from tqdm import tqdm
import gc
from typing import Any
import torch.nn.functional as F


def find_matching_indices(old, new):
    # Find the starting common sequence
    start_common = 0
    for i, j in zip(old, new):
        if i == j:
            start_common += 1
        else:
            break

    # Find the ending common sequence
    end_common_old = len(old) - 1
    end_common_new = len(new) - 1
    while end_common_old >= start_common and end_common_new >= start_common:
        if old[end_common_old] == new[end_common_new]:
            end_common_old -= 1
            end_common_new -= 1
        else:
            break

    return list(range(start_common)) + list(range(end_common_old + 1, len(old))), \
           list(range(start_common)) + list(range(end_common_new + 1, len(new)))
       
           
def get_ca_layers(unet, with_to_k=True):

    sub_nets = unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ## get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
    
    return projection_matrices, ca_layers, og_matrices


def prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, test_set, 
                tokenizer, with_to_k=True, all_words=False, prepare_k_v_for_lora=False):
    
    with torch.no_grad():
        all_contexts, all_valuess = [], []
        
        for curr_item in test_set:
            gc.collect()
            torch.cuda.empty_cache()
            
            #### restart LDM parameters
            num_ca_clip_layers = len(ca_layers)
            for idx_, l in enumerate(ca_layers):
                l.to_v = copy.deepcopy(og_matrices[idx_])
                projection_matrices[idx_] = l.to_v
                if with_to_k:
                    l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                    projection_matrices[num_ca_clip_layers + idx_] = l.to_k
            
            old_embs, new_embs = [], []
            extended_old_indices, extended_new_indices = [], []
            
            #### indetify corresponding destinations for each token in old_emb
            # Bulk tokenization
            texts_old = [item[0] for item in curr_item["old"]]
            texts_new = [item[0] for item in curr_item["new"]]
            texts_combined = texts_old + texts_new

            tokenized_inputs = tokenizer(
                texts_combined,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Text embeddings
            text_embeddings = text_encoder(tokenized_inputs.input_ids.to(text_encoder.device))[0]
            old_embs.extend(text_embeddings[:len(texts_old)])
            new_embs.extend(text_embeddings[len(texts_old):])

            # Find matching indices
            for old_text, new_text in zip(texts_old, texts_new):
                tokens_a = tokenizer(old_text).input_ids
                tokens_b = tokenizer(new_text).input_ids
                
                old_indices, new_indices = find_matching_indices(tokens_a, tokens_b)
                
                if old_indices[-1] >= new_indices[-1]:
                    extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77)))
                    extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77 - (old_indices[-1] - new_indices[-1]))))
                else:
                    extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77)))
                    extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77 - (new_indices[-1] - old_indices[-1]))))

            #### prepare batch: for each pair of setences, old context and new values
            contexts, valuess = [], []
            if not all_words:
                for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                    context = old_emb[extended_old_indices[idx]].detach()
                    values = []
                    for layer in projection_matrices:
                        values.append(layer(new_emb[extended_new_indices[idx]]).detach())
                    contexts.append(context)
                    valuess.append(values)
            
                all_contexts.append(contexts)
                all_valuess.append(valuess)
            else:
                if prepare_k_v_for_lora:
                    # prepare for lora, then no need to use new_emb
                    for idx, old_emb in enumerate(old_embs):
                        context = old_emb.detach()
                        values = []
                        for layer in projection_matrices:
                            values.append(layer(old_emb).detach())
                        contexts.append(context)
                        valuess.append(values)
                else:
                    # need to use new_emb
                    for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                        context = old_emb.detach()
                        values = []
                        for layer in projection_matrices:
                            values.append(layer(new_emb).detach())
                        contexts.append(context)
                        valuess.append(values)
            
                all_contexts.append(contexts)
                all_valuess.append(valuess)
        
        return all_contexts, all_valuess
            
            
def closed_form_refinement(
    projection_matrices,
    all_contexts=None,
    all_valuess=None,
    lamb=0.5,
    preserve_scale=1,
    cache_dict=None,
    cache_dict_path=None,
    cache_mode=False,
    cfr_args=None,
    original_weights=None,
    weight_names=None,
    selft_masks=None,
    projection_matrix=None,
    projection_max_iters=1,
    projection_convergence_tol=None,
):
    
    with torch.no_grad():
        if cache_dict_path is not None:
            cache_dict = torch.load(cache_dict_path, map_location=projection_matrices[0].weight.device)
        cache_dict = cache_dict if cache_dict is not None else {}
            
        l1_weight = getattr(cfr_args, "l1sp_weight", 0.0) if cfr_args is not None else 0.0
        l2_weight = getattr(cfr_args, "l2sp_weight", 0.0) if cfr_args is not None else 0.0
        max_projection_iters = getattr(cfr_args, "projection_max_iters", projection_max_iters) if cfr_args is not None else projection_max_iters
        if max_projection_iters is None or max_projection_iters < 1:
            max_projection_iters = 1
        convergence_tol = getattr(cfr_args, "projection_convergence_tol", projection_convergence_tol) if cfr_args is not None else projection_convergence_tol
        if convergence_tol is not None:
            convergence_tol = float(convergence_tol)
        convergence_eps = getattr(cfr_args, "projection_convergence_eps", 1e-12) if cfr_args is not None else 1e-12
        apply_selft = selft_masks is not None

        print(f"Using CFR with l1_weight: {l1_weight}, l2_weight: {l2_weight}, use_projection: {projection_matrix is not None}, max_projection_iters: {max_projection_iters}, convergence_tol: {convergence_tol}, apply_selft: {apply_selft}")
        for layer_num in tqdm(range(len(projection_matrices))):
            gc.collect()
            torch.cuda.empty_cache()

            # Pre-compute layer-specific constants
            weight_tensor = projection_matrices[layer_num].weight
            mat_identity = torch.eye(
                weight_tensor.shape[1],
                device=weight_tensor.device,
                dtype=weight_tensor.dtype,
            )
            lambda_identity = lamb * mat_identity

            base_total_for_mat1 = torch.zeros_like(weight_tensor)
            base_total_for_mat2 = torch.zeros_like(mat_identity)
            base_total_values_norm = torch.zeros(
                (), device=weight_tensor.device, dtype=weight_tensor.dtype
            )

            if all_contexts is not None and all_valuess is not None:
                for contexts, valuess in zip(all_contexts, all_valuess):
                    # Convert contexts (QK^T) and values to tensors
                    contexts_tensor = torch.stack(contexts, dim=2)
                    values_tensor = torch.stack([vals[layer_num] for vals in valuess], dim=2)
                    
                    # Aggregate sums for mat1, mat2 using matrix multiplication
                    for_mat1 = torch.bmm(values_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0) #QK^T V
                    for_mat2 = torch.bmm(contexts_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0) # (QK^T)^2
                    
                    base_total_for_mat1 += for_mat1
                    base_total_for_mat2 += for_mat2
                    base_total_values_norm += (values_tensor * values_tensor).sum()

                del for_mat1, for_mat2
                
            if cache_mode: 
                # cache the results
                key_mat1 = f'{layer_num}_for_mat1'
                key_mat2 = f'{layer_num}_for_mat2'
                key_norm = f'{layer_num}_values_norm'
                if cache_dict.get(key_mat1, None) is None:
                    cache_dict[key_mat1] = base_total_for_mat1
                    cache_dict[key_mat2] = base_total_for_mat2
                    cache_dict[key_norm] = base_total_values_norm
                else:
                    cache_dict[key_mat1] += base_total_for_mat1
                    cache_dict[key_mat2] += base_total_for_mat2
                    existing_norm = torch.as_tensor(
                        cache_dict.get(key_norm, 0.0),
                        device=base_total_values_norm.device,
                        dtype=base_total_values_norm.dtype,
                    )
                    cache_dict[key_norm] = existing_norm + base_total_values_norm
            else:
                # CFR calculation
                if cache_dict_path is not None or cache_dict is not None:
                    # Stored QK^V activations from preserve concepts
                    cached_mat1 = cache_dict.get(f'{layer_num}_for_mat1', 0.0)
                    cached_mat2 = cache_dict.get(f'{layer_num}_for_mat2', 0.0)
                    cached_norm = cache_dict.get(f'{layer_num}_values_norm', 0.0)
                    cached_mat1 = torch.as_tensor(
                        cached_mat1, device=weight_tensor.device, dtype=weight_tensor.dtype
                    )
                    cached_mat2 = torch.as_tensor(
                        cached_mat2, device=weight_tensor.device, dtype=weight_tensor.dtype
                    )
                    cached_norm = torch.as_tensor(
                        cached_norm, device=weight_tensor.device, dtype=weight_tensor.dtype
                    )
                    base_total_for_mat1 += preserve_scale * cached_mat1
                    base_total_for_mat2 += preserve_scale * cached_mat2
                    base_total_values_norm += preserve_scale * cached_norm
                
                # Layer naming and original weight lookup (cached per layer)
                layer_name = None
                if weight_names is not None and layer_num < len(weight_names):
                    layer_name = weight_names[layer_num]

                layer_orig_weight = None
                if original_weights is not None:
                    if isinstance(original_weights, dict):
                        key = layer_name
                        if key is not None:
                            layer_orig_weight = original_weights.get(key, None)
                    else:
                        layer_orig_weight = original_weights[layer_num]

                proj = None
                if projection_matrix is not None:
                    if isinstance(projection_matrix, dict):
                        if layer_name is None and weight_names is not None and layer_num < len(weight_names):
                            layer_name = weight_names[layer_num]
                        if layer_name is not None:
                            proj = projection_matrix.get(layer_name, None)
                    else:
                        proj = projection_matrix

                if proj is not None:
                    proj = proj.to(weight_tensor.device, weight_tensor.dtype)

                mask_tensor = None
                if apply_selft:
                    if layer_name is None and weight_names is not None and layer_num < len(weight_names):
                        layer_name = weight_names[layer_num]
                    if isinstance(selft_masks, dict) and layer_name in selft_masks:
                        mask_tensor = selft_masks[layer_name]

                effective_iters = max_projection_iters if proj is not None else 1
                current_weight = weight_tensor.detach().clone()
                base_error_available = (
                    base_total_values_norm.item() > 0
                    or base_total_for_mat1.abs().sum().item() > 0
                )

                def compute_mapping_error(weight):
                    term1 = torch.sum((weight @ base_total_for_mat2) * weight)
                    term2 = torch.sum(weight * base_total_for_mat1)
                    err_sq = term1 - 2 * term2 + base_total_values_norm
                    return torch.sqrt(torch.clamp(err_sq, min=0.0))

                baseline_mapping_error = None
                if base_error_available:
                    baseline_mapping_error = compute_mapping_error(current_weight).detach()
                
                progress_bar = tqdm(range(effective_iters), desc=f"Layer {layer_num} Iterations")
                for iter_idx in progress_bar:
                    total_for_mat1 = base_total_for_mat1 + lamb * current_weight
                    total_for_mat2 = base_total_for_mat2 + lambda_identity

                    if l2_weight > 0.0 and layer_orig_weight is not None:
                        orig_tensor = layer_orig_weight.to(total_for_mat1.device, total_for_mat1.dtype)
                        total_for_mat1 = total_for_mat1 + l2_weight * orig_tensor
                        total_for_mat2 = total_for_mat2 + l2_weight * mat_identity.to(
                            total_for_mat2.device, total_for_mat2.dtype
                        )

                    # Solve for the updated weight matrix
                    try:
                        solved_weight_t = torch.linalg.solve(total_for_mat2.T, total_for_mat1.T)
                        updated_weight = solved_weight_t.T
                    except RuntimeError:
                        updated_weight = total_for_mat1 @ torch.linalg.pinv(total_for_mat2)

                    # Apply geometry-aware projection if provided
                    if proj is not None:
                        if layer_orig_weight is not None:
                            base_weight = layer_orig_weight.to(updated_weight.device, updated_weight.dtype)
                        else:
                            base_weight = current_weight.to(updated_weight.device, updated_weight.dtype)

                        delta = updated_weight - base_weight
                        projected_delta = delta @ proj
                        updated_weight = base_weight + projected_delta

                    # Apply L1-SP via soft-thresholding relative to original weights
                    if l1_weight > 0.0 and layer_orig_weight is not None:
                        orig = layer_orig_weight.to(updated_weight.device, updated_weight.dtype)
                        delta = updated_weight - orig
                        shrink = torch.clamp(delta.abs() - l1_weight, min=0.0)
                        shrink = shrink * delta.sign()
                        updated_weight = orig + shrink

                    # Apply SelFT masks to preserve important parameters
                    if mask_tensor is not None:
                        mask = mask_tensor.to(updated_weight.device, dtype=updated_weight.dtype)
                        base_weight_for_mask = (
                            layer_orig_weight.to(updated_weight.device, updated_weight.dtype)
                            if layer_orig_weight is not None
                            else current_weight.to(updated_weight.device, updated_weight.dtype)
                        )
                        updated_weight = mask * updated_weight + (1.0 - mask) * base_weight_for_mask

                    # Check convergence before preparing for next iteration
                    weight_delta = updated_weight - current_weight.to(updated_weight.device, updated_weight.dtype)
                    delta_norm = torch.norm(weight_delta)
                    reference_norm = torch.norm(current_weight.to(updated_weight.device, updated_weight.dtype))
                    if reference_norm < convergence_eps:
                        reference_norm = convergence_eps
                    relative_change = (delta_norm / reference_norm).item()

                    current_weight = updated_weight.detach().to(weight_tensor.device, weight_tensor.dtype)

                    if convergence_tol is not None:
                        if base_error_available and baseline_mapping_error is not None:
                            mapping_error = compute_mapping_error(current_weight).detach()
                            baseline_value = baseline_mapping_error.item()
                            if baseline_value < convergence_eps:
                                metric_value = mapping_error.item()
                            else:
                                metric_value = mapping_error.item() / max(baseline_value, convergence_eps)
                            progress_bar.set_postfix({"metric": metric_value})
                            if metric_value <= convergence_tol:
                                break
                        elif relative_change <= convergence_tol:
                            break
                    progress_bar.set_postfix({"threshold": convergence_tol})
                    
                projection_matrices[layer_num].weight.data = current_weight.to(
                    projection_matrices[layer_num].weight.data.dtype
                )
                
                del base_total_for_mat1, base_total_for_mat2, base_total_values_norm



def importance_sampling_fn(t, temperature=0.05):
    """Importance Sampling Function f(t)"""
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))
        
        
class AttnController:
    def __init__(self) -> None:
        self.attn_probs = []
        self.logs = []
        
    def __call__(self, attn_prob, m_name, preserve_prior, latent_num) -> Any:
        bs, _ = self.concept_positions.shape
        
        if preserve_prior:
            attn_prob = attn_prob[:attn_prob.shape[0] // latent_num]
            
        if self.use_gsam_mask:
            d = int(attn_prob.shape[1] ** 0.5)
            resized_mask = F.interpolate(self.mask, size=(d, d), mode='nearest')
            
            # # save mask
            # img_array = (resized_mask > 0.5).to(torch.uint8) * 255
            # from PIL import Image
            # img = Image.fromarray(img_array[0][0].cpu().numpy())
            # img.save('./sam_outputs/bool_image.png')
            
            resized_mask = (resized_mask > 0.5).view(-1)
            attn_prob = attn_prob[:, resized_mask, :]
            target_attns = attn_prob[:, :, self.concept_positions[0]]
        else:
            head_num = attn_prob.shape[0] // bs
            target_attns = attn_prob.masked_select(self.concept_positions[:,None,:].repeat(head_num, 1, 1)).reshape(-1, self.concept_positions[0].sum())
        
        self.attn_probs.append(target_attns)
        self.logs.append(m_name)
        
    def set_concept_positions(self, concept_positions, mask=None, use_gsam_mask=False):
        self.concept_positions = concept_positions
        self.mask = mask
        self.use_gsam_mask = use_gsam_mask
        
    def loss(self):
        return sum(torch.norm(item) for item in self.attn_probs)
        
    def zero_attn_probs(self):
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None
            
            
def prompt_augmentation(content, augment=True, sampled_indices=None, concept_type='object'):
    if augment:
        # some sample prompts provided
        if concept_type == 'object':
            prompts = [
                # object augmentation
                ("{} in a photo".format(content), content),
                ("{} in a snapshot".format(content), content),
                ("A snapshot of {}".format(content), content),
                ("A photograph showcasing {}".format(content), content),
                ("An illustration of {}".format(content), content),
                ("A digital rendering of {}".format(content), content),
                ("A visual representation of {}".format(content), content),
                ("A graphic of {}".format(content), content),
                ("A shot of {}".format(content), content),
                ("A photo of {}".format(content), content),
                ("A black and white image of {}".format(content), content),
                ("A depiction in portrait form of {}".format(content), content),
                ("A scene depicting {} during a public gathering".format(content), content),
                ("{} captured in an image".format(content), content),
                ("A depiction created with oil paints capturing {}".format(content), content),
                ("An image of {}".format(content), content),
                ("A drawing capturing the essence of {}".format(content), content),
                ("An official photograph featuring {}".format(content), content),
                ("A detailed sketch of {}".format(content), content),
                ("{} during sunset/sunrise".format(content), content),
                ("{} in a detailed portrait".format(content), content),
                ("An official photo of {}".format(content), content),
                ("Historic photo of {}".format(content), content),
                ("Detailed portrait of {}".format(content), content),
                ("A painting of {}".format(content), content),
                ("HD picture of {}".format(content), content),
                ("Magazine cover capturing {}".format(content), content),
                ("Painting-like image of {}".format(content), content),
                ("Hand-drawn art of {}".format(content), content),
                ("An oil portrait of {}".format(content), content),
                ("{} in a sketch painting".format(content), content),
            ]
            
        elif concept_type == 'style':
            # art augmentation
            prompts = [
                ("An artwork by {}".format(content), content),
                ("Art piece by {}".format(content), content),
                ("A recent creation by {}".format(content), content),
                ("{}'s renowned art".format(content), content),
                ("Latest masterpiece by {}".format(content), content),
                ("A stunning image by {}".format(content), content),
                ("An art in {}'s style".format(content), content),
                ("Exhibition artwork of {}".format(content), content),
                ("Art display by {}".format(content), content),
                ("a beautiful painting by {}".format(content), content),
                ("An image inspired by {}'s style".format(content), content),
                ("A sketch by {}".format(content), content),
                ("Art piece representing {}".format(content), content),
                ("A drawing by {}".format(content), content),
                ("Artistry showcasing {}".format(content), content),
                ("An illustration by {}".format(content), content),
                ("A digital art by {}".format(content), content),
                ("A visual art by {}".format(content), content),
                ("A reproduction inspired by {}'s colorful, expressive style".format(content), content),
                ("Famous painting of {}".format(content), content),
                ("A famous art by {}".format(content), content),
                ("Artistic style of {}".format(content), content),
                ("{}'s famous piece".format(content), content),
                ("Abstract work of {}".format(content), content),
                ("{}'s famous drawing".format(content), content),
                ("Art from {}'s early period".format(content), content),
                ("A portrait by {}".format(content), content),
                ("An imitation reflecting the style of {}".format(content), content),
                ("An painting from {}'s collection".format(content), content),
                ("Vibrant reproduction of artwork by {}".format(content), content),
                ("Artistic image influenced by {}".format(content), content),
            ] 
        else:
            raise ValueError("unknown concept type.")
    else: 
        prompts = [
            ("A photo of {}".format(content), content),
        ]
    
    if sampled_indices is not None:
        sampled_prompts = [prompts[i] for i in sampled_indices if i < len(prompts)]
    else:
        sampled_prompts = prompts
        
    return sampled_prompts
   
