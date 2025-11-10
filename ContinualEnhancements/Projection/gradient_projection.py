import os
import torch
import torch.nn.functional as F
import openai
import re
import ast

def build_projection_matrix(filtered_embedding_matrix, device):
    """
    Construct the orthogonal projection matrix used for gradient/weight projection.

    Args:
        filtered_embedding_matrix: Anchor embedding matrix [embedding_dim, num_anchors]
        device: Target device for computations

    Returns:
        Projection matrix P with shape [embedding_dim, embedding_dim] or None if no anchors provided.
    """
    if filtered_embedding_matrix is None or filtered_embedding_matrix.shape[1] == 0:
        return None

    # Use float32 for stable linear algebra (bf16 lacks eig implementation)
    C = filtered_embedding_matrix.to(device=device, dtype=torch.float32)
    embedding_dim = C.shape[0]
    num_anchors = C.shape[1]

    try:
        # Compute C^T C
        CTC = C.T @ C  # [num_anchors, num_anchors]

        # Regularization term (scaled identity matrix) is added to ensure numerical stability
        reg_strength = max(1e-4, 1e-6 * torch.trace(CTC).item() / num_anchors)
        reg_term = reg_strength * torch.eye(num_anchors, device=device, dtype=CTC.dtype)
        
        # Invertible matrix has real and positive eigenvalues
        eigenvals = torch.linalg.eigvals(CTC + reg_term).real
        
        # Check condition number for numerical stability
        condition_number = eigenvals.max() / eigenvals.min()
        
        # If matrix is too sensitive to small changes, matrix is ill-conditioned
        if condition_number > 1e10:
            raise torch.linalg.LinAlgError("Matrix is ill-conditioned")

        # Invert C^T C with regularization term
        CTC_inv = torch.linalg.inv(CTC + reg_term)
        
        # Orthogonal Projection matrix P = I - C @ CTC_inv @ C.T
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ CTC_inv @ C.T  # [embedding_dim, embedding_dim]
        
    except torch.linalg.LinAlgError:
        # Fallback to pseudoinverse if matrix is singular
        print(f"Warning: Using pseudoinverse for gradient projection")

        # Compute Moore-Penrose pseudoinverse of C^T
        C_pinv = torch.linalg.pinv(C.T)  # [num_anchors, embedding_dim]

        # Projection matrix P = I - C @ C_pinv
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ C_pinv
    
    return P


def apply_gradient_projection(model, filtered_embedding_matrix, device, accelerator=None):
    """
    Apply gradient projection using pre-filtered embedding matrix.
    
    Args:
        model: The UNet model
        filtered_embedding_matrix: Pre-computed and filtered embedding matrix [embedding_dim, num_anchors]
        device: Device to run computations on
        accelerator: HuggingFace accelerator (optional)
    """
    projection_matrix = build_projection_matrix(filtered_embedding_matrix, device)
    if projection_matrix is None:
        return  # No projection needed if no anchor embeddings
    proj_dtype = projection_matrix.dtype
    
    C = filtered_embedding_matrix.to(device=device, dtype=proj_dtype)  # [embedding_dim, num_anchors]
    embedding_dim = C.shape[0]
    
    # Get model with gradients
    model_to_process = accelerator.unwrap_model(model) if accelerator else model

    # Build name->parameter dict once (for pairing lora_up/lora_down)
    name_to_param = dict(model_to_process.named_parameters())
    processed_lora = set()  # to avoid double-processing of the same lora pair

    # Iterate through all K and V weights in cross-attention layers    
    for name, param in model_to_process.named_parameters():
        if (param.grad is not None and 
            'attn2' in name and 
            ('to_k' in name or 'to_v' in name)):
            
            # LoRA processing
            if "lora_layer" in name and param.grad.shape[1] != embedding_dim:
                if not name.endswith(".lora_layer.down.weight"):
                    continue  # Process only down weights (We'll find up weights in the same loop)
                
                prefix = name.rsplit(".lora_layer.down.weight", 1)[0]
                if prefix in processed_lora:
                    continue  # Already processed this LoRA pair
                processed_lora.add(prefix)
                
                A_name = prefix + ".lora_layer.up.weight"
                B_name = prefix + ".lora_layer.down.weight"
                A = name_to_param.get(A_name, None)
                B = name_to_param.get(B_name, None)
                if A is None or B is None or A.grad is None or B.grad is None:
                    print(f"Warning: Could not find LoRA pair for {name}, skipping projection")
                    continue
                
                # Reconstruct Geff (d_out x d_in), project on input space, then decompose back
                scale = 1.0
                Geff = _reconstruct_Geff_from_lora(A.data, B.data, A.grad.data.to(C.dtype), B.grad.data.to(C.dtype), reg=1e-4, scale=scale)
                if Geff is None or Geff.shape[1] != embedding_dim:
                    print(f"Warning: Could not reconstruct Geff for {name}, defaulting to down matrix projection")
                    gB = B.grad
                    if gB.ndim == 2 and gB.shape[1] == embedding_dim:
                        B.grad.data = (gB.to(projection_matrix.dtype) @ projection_matrix).to(gB.dtype)
                    continue
                
                Geff_proj = (Geff.to(proj_dtype) @ projection_matrix) # right projection in text dimension
                gA_new, gB_new = _decompose_Geff_to_lora_grads(Geff_proj.to(A.grad.dtype), A.data, B.data, scale=scale)
                
                # Write back new grads
                A.grad.data = gA_new.to(A.grad.dtype)
                B.grad.data = gB_new.to(B.grad.dtype)
                
                # Done for this LoRA pair
                continue
                
            # Original projection behavior (Non-LoRA)
            original_shape = param.grad.shape
            if len(original_shape) == 2:
                # Standard linear layer: [out_features, in_features]
                grad_matrix = param.grad.to(proj_dtype)  # [out_features, in_features]
                projected_grad = grad_matrix @ projection_matrix  # Project along input dimension               
                param.grad.data = projected_grad.to(param.grad.dtype)
            elif len(original_shape) == 1:
                # Skip bias terms - no projection needed
                continue
            else:
                # Handle unexpected shapes by reshaping
                grad_reshaped = param.grad.view(original_shape[0], -1).to(proj_dtype)
                projected_grad = grad_reshaped @ projection_matrix
                param.grad.data = projected_grad.view(original_shape).to(param.grad.dtype)

# --- Minimal helpers for LoRA ΔW reconstruction/decomposition ---
def _reconstruct_Geff_from_lora(A, B, gA, gB, reg=1e-4, scale=1.0):
    """
    Reconstruct an estimate of full-space grad Geff (d_out x d_in) for ΔW = scale * A @ B
    using small rxr solves; combine two estimates for stability.
    """
    d_out, r = A.shape
    r2, d_in = B.shape
    if r != r2:
        return None

    # (1) From gA = scale * Geff * B^T  => Geff ≈ (1/scale) * gA @ (B^T)^+
    # Use (B^T)^+ = (B B^T + λI)^(-1) @ B
    BBt = (B @ B.T) + reg * torch.eye(r, device=B.device, dtype=B.dtype)  # (r, r)
    gA_small = gA @ torch.linalg.solve(BBt, torch.eye(r, device=B.device, dtype=B.dtype))  # (d_out, r)
    Geff_a = (gA_small @ B) / max(scale, 1e-12)  # (d_out, d_in)

    # (2) From gB = scale * A^T * Geff  => Geff ≈ (1/scale) * (A^T)^+ @ gB
    # Use (A^T)^+ = A @ (A^T A + λI)^(-1)
    AtA = (A.T @ A) + reg * torch.eye(r, device=A.device, dtype=A.dtype)  # (r, r)
    Geff_b = (A @ torch.linalg.solve(AtA, torch.eye(r, device=A.device, dtype=A.dtype)) @ gB) / max(scale, 1e-12)

    return 0.5 * (Geff_a + Geff_b)

def _decompose_Geff_to_lora_grads(Geff_proj, A, B, scale=1.0):
    """Map projected full-space grad back to LoRA grads."""
    gA_new = scale * (Geff_proj @ B.T)   # (d_out, r)
    gB_new = scale * (A.T @ Geff_proj)   # (r, d_in)
    return gA_new, gB_new
    
def get_anchor_embeddings(anchor_prompts, text_encoder, tokenizer, device):
    """
    Encode anchor prompts into text embeddings, extracting concept tokens after format prefixes.
    
    Args:
        anchor_prompts (list): List of anchor prompts to encode.
        text_encoder: The text encoder model.
        tokenizer: The tokenizer for the text encoder.
        device: Device to run computations on (e.g., 'cuda' or 'cpu').
    """    
    if len(anchor_prompts) > 0:
        print(f"Encoding {len(anchor_prompts)} anchor prompts...")
        anchor_embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(anchor_prompts), batch_size):
            batch_prompts = anchor_prompts[i:i+batch_size]
            
            with torch.no_grad():
                for prompt in batch_prompts:
                    # Tokenize individual prompt
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    
                    # Get text embeddings
                    text_embeddings = text_encoder(text_inputs.input_ids)[0]
                    
                    # Extract concept embedding after prefix
                    concept_embedding = extract_concept_tokens_after_prefix(
                        text_embeddings.unsqueeze(0), text_inputs, tokenizer
                    )
                    
                    anchor_embeddings.append(concept_embedding.cpu())
        
        print(f"Successfully encoded {len(anchor_embeddings)} anchor embeddings")
        
        print(f"Now filtering to remove redundant embeddings...")
        
        # Stack into matrix [embedding_dim, num_anchors]
        C = torch.stack(anchor_embeddings, dim=1)
        
        # Remove similar embeddings
        C_filtered = remove_similar_embeddings(C)
        
        print(f"Final anchor embedding matrix shape: {C_filtered.shape}")
        
        return C_filtered
    else:
        raise ValueError("No anchor prompts provided for encoding.")

def extract_concept_tokens_after_prefix(text_embeddings, text_inputs, tokenizer, verbose=False):
    """
    Extract embeddings for tokens after format prefixes and average them if multiple.
    
    Args:
        text_embeddings: Full sequence embeddings [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim]
        text_inputs: Tokenized inputs
        tokenizer: The tokenizer used
        
    Returns:
        Tensor: Averaged embedding for concept tokens
    """
    # Handle different tensor shapes
    if len(text_embeddings.shape) == 2:
        # Shape is [seq_len, hidden_dim] - add batch dimension
        text_embeddings = text_embeddings.unsqueeze(0)
        seq_len, hidden_dim = text_embeddings.shape[1], text_embeddings.shape[2]
        batch_size = 1
    elif len(text_embeddings.shape) == 3:
        # Shape is [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = text_embeddings.shape
    elif len(text_embeddings.shape) == 4:
        # Shape is [batch_size, extra_dim, seq_len, hidden_dim] - squeeze extra dimension
        original_shape = text_embeddings.shape
        text_embeddings = text_embeddings.squeeze(1)  # Remove second dimension
        batch_size, seq_len, hidden_dim = text_embeddings.shape
    else:
        raise ValueError(f"Unexpected text_embeddings shape: {text_embeddings.shape}")
    
    # Convert input_ids back to tokens
    tokens = tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])
    
    # Define prefixes to look for
    style_prefix_tokens = tokenizer.tokenize("an image in the style of")
    object_prefix_tokens = tokenizer.tokenize("an image of a")
    object_prefix_tokens_alt = tokenizer.tokenize("an image of")  # Without "a"
    
    # Find where concept tokens start
    concept_start_idx = None
    
    # Check for style prefix
    if find_token_sequence(tokens, style_prefix_tokens) is not None:
        prefix_end = find_token_sequence(tokens, style_prefix_tokens) + len(style_prefix_tokens)
        concept_start_idx = prefix_end
        
    # Check for object prefix (with "a")
    elif find_token_sequence(tokens, object_prefix_tokens) is not None:
        prefix_end = find_token_sequence(tokens, object_prefix_tokens) + len(object_prefix_tokens)
        concept_start_idx = prefix_end
        
    # Check for object prefix (without "a")
    elif find_token_sequence(tokens, object_prefix_tokens_alt) is not None:
        prefix_end = find_token_sequence(tokens, object_prefix_tokens_alt) + len(object_prefix_tokens_alt)
        concept_start_idx = prefix_end
    
    # If no prefix found, use last meaningful token
    if concept_start_idx is None:
        attention_mask = text_inputs.attention_mask[0]
        print(f"Sequence Length: {seq_len}, Attention Mask Length: {attention_mask.sum().item()}")
        last_token_idx = min(attention_mask.sum().item() - 2, seq_len - 1)
        last_token_idx = max(0, last_token_idx)  # Ensure non-negative
        print(f"Warning: Could not find expected prefix in tokens: '{tokens}'")
        print(f"Using last meaningful token at index {last_token_idx} '{tokens[last_token_idx]}' as fallback")
        return text_embeddings[0, last_token_idx, :]
    
    # Find end of meaningful tokens (before padding)
    attention_mask = text_inputs.attention_mask[0]
    last_meaningful_idx = min(attention_mask.sum().item() - 1, seq_len - 1)
    
    # Ensure concept_start_idx is within bounds
    concept_start_idx = min(concept_start_idx, seq_len - 1)
    concept_start_idx = max(0, concept_start_idx)
    
    # Extract concept token indices with bounds checking
    concept_end_idx = min(last_meaningful_idx + 1, seq_len)  # Don't exceed tensor length
    concept_indices = list(range(concept_start_idx, concept_end_idx))
    
    # Additional bounds checking and filtering
    valid_concept_indices = []
    for idx in concept_indices:
        # Check bounds
        if 0 <= idx < seq_len and idx < len(tokens):
            # Check if not special token
            if not tokens[idx].startswith('[') and tokens[idx] != '<|endoftext|>' and tokens[idx] != '<pad>':
                valid_concept_indices.append(idx)
    
    concept_indices = valid_concept_indices
    
    # If no valid concept tokens found, fallback to last token
    if not concept_indices:
        attention_mask = text_inputs.attention_mask[0]
        last_token_idx = min(attention_mask.sum().item() - 2, seq_len - 1)
        last_token_idx = max(0, last_token_idx)
        print(f"Warning: No valid concept tokens found after prefix in: '{tokens}'")
        print(f"Using last meaningful token at index {last_token_idx} '{tokens[last_token_idx]}' as fallback")
        return text_embeddings[0, last_token_idx, :]

    # Ensure indices are within bounds one more time
    concept_indices = [idx for idx in concept_indices if 0 <= idx < seq_len]
    
    if not concept_indices:
        print(f"Error: All concept indices out of bounds!")
        return text_embeddings[0, 0, :]  # Use first token
    
    # Extract and average concept embeddings
    try:
        concept_embeddings = text_embeddings[0, concept_indices, :]  # [num_concept_tokens, hidden_dim]
        
        # Average if multiple tokens
        if len(concept_indices) > 1:
            if verbose: print(f"Averaged concept tokens: {[tokens[i] for i in concept_indices]} ({len(concept_indices)} tokens)")
            return concept_embeddings.mean(dim=0)
        else:
            if verbose: print(f"Extracted concept tokens: {[tokens[i] for i in concept_indices]} ({len(concept_indices)} tokens)")
            return concept_embeddings[0]
            
    except Exception as e:
        print(f"Error extracting concept embeddings: {e}")
        print(f"  concept_indices: {concept_indices}")
        print(f"  text_embeddings.shape: {text_embeddings.shape}")
        return text_embeddings[0, 0, :]
    
def find_token_sequence(tokens, target_sequence):
    """
    Find the starting index of a token sequence in the full token list.
    Returns None if not found.
    """
    if len(target_sequence) == 0:
        return None
        
    # Normalize tokens for comparison (handle BPE markers, case, etc.)
    tokens_clean = [t.lower().replace('##', '').replace('▁', '') for t in tokens]
    target_clean = [t.lower().replace('##', '').replace('▁', '') for t in target_sequence]
    
    for i in range(len(tokens_clean) - len(target_clean) + 1):
        if tokens_clean[i:i+len(target_clean)] == target_clean:
            return i
    
    return None

def remove_similar_embeddings(C, similarity_threshold=0.95):
    """
    Remove embeddings that are too similar to avoid rank deficiency.
    
    Args:
        C: Embedding matrix [embedding_dim, num_anchors]
        similarity_threshold: Remove embeddings with cosine similarity > threshold
        
    Returns:
        C_filtered: Matrix with similar embeddings removed
    """
    if C.shape[1] <= 1:
        return C
    
    # Normalize embeddings for cosine similarity
    C_norm = C / (C.norm(dim=0, keepdim=True) + 1e-8)
    
    # Compute pairwise cosine similarities
    similarities = C_norm.T @ C_norm  # [num_anchors, num_anchors]
    
    # Find embeddings to keep (greedy selection)
    num_anchors = C.shape[1]
    keep_indices = [0]  # Always keep the first one
    
    for i in range(1, num_anchors):
        # Check if this embedding is too similar to any we're keeping
        max_similarity = similarities[i, keep_indices].max().item()
        
        if max_similarity < similarity_threshold:
            keep_indices.append(i)
    
    print(f"Keeping {len(keep_indices)}/{num_anchors} anchor embeddings after similarity filtering")
    
    return C[:, keep_indices]

def generate_gradient_projection_prompts(file_path, num_prompts, concept_type, previously_unlearned, target_concept_list, dual_domain=True):
    """
    Generate anchor prompts using gpt and write to file_path.
    Args:
        file_path (str): Path to save the generated anchor prompts.
        num_prompts (int): Number of prompts to generate.
        concept_type (str): Type of concept to generate prompts for.
        previously_unlearned (list): List of previously unlearned concepts.
        target_concept_list (list): Current concepts to unlearn
    """
    # Get openai API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Create list of concepts to exclude from prompt generation
    # Previously Unlearned Concepts (continual)
    if previously_unlearned and "[" in previously_unlearned:
        previously_unlearned = ast.literal_eval(previously_unlearned)
    else:
        previously_unlearned = [previously_unlearned] if previously_unlearned else []
        
    # Concepts we're unlearning this training run
    no_duplicates = list(dict.fromkeys(target_concept_list))
    if len(no_duplicates) < len(target_concept_list):
        print(f"Removed {len(target_concept_list) - len(no_duplicates)} duplicate concepts from '{target_concept_list}' to '{no_duplicates}'")
        target_concept_list = no_duplicates

    for i in range(len(target_concept_list)):
        target_concept_list[i] = target_concept_list[i].replace(" Style", "")
        target_concept_list[i] = target_concept_list[i].replace("An image of ", "")
        print(f"Excluding for gradient projection prompts: '{target_concept_list[i]}'")

    # Generate style prompts
    print(f"\n=== Generating Style Prompts ===")
    style_prompts = []

    # Add previously unlearned styles to prompt collection
    if concept_type == "style":
        for concept in previously_unlearned:
            if concept == "":
                continue
            prompt = concept.replace("_", " ")
            prompt = f"An image in the style of {prompt}"
            print(f"Adding previously unlearned concept to prompt: '{prompt}'")
            style_prompts.append(prompt)
        print(f"Added '{len(style_prompts)}' previously unlearned style prompts")
    
    # Ensure gpt doesn't generate prompts that we plan on unlearning this training run
    style_exclusions = target_concept_list if concept_type == "style" else []
    style_messages = [
        {"role": "system", "content": "You are an expert at generating creative image captions. You will generate random prompts in the format 'An image in the style of {style}'. Remove unnecessary suffixes like 'style' or 'art' from style names. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line."},
        {"role": "user", "content": f'Generate {num_prompts} random prompts for images in the format "An image in the style of {{style}}". The styles should be diverse art styles, artist names, or artistic movements. None of the styles should contain the words: {style_exclusions}.' if style_exclusions else f'Generate {num_prompts} random prompts for images in the format "An image in the style of {{style}}". The styles should be diverse art styles, artist names, or artistic movements.'},
    ]
    
    if concept_type == "style" or dual_domain:
        print(f"Style GPT Query: '{style_messages[1]['content']}'")
        if dual_domain:
            num_style_prompts = (num_prompts // 2) - len(style_prompts)
        else:
            num_style_prompts = num_prompts - len(style_prompts)
        style_prompts.extend(generate_prompts_with_gpt(style_messages, num_style_prompts, style_exclusions))
    else:
        print(f"Skipping style prompt generation for concept type '{concept_type}'")
    
    # Generate object prompts  
    print(f"\n=== Generating Object Prompts ===")
    object_prompts = []

    # Add previously unlearned objects to prompt collection
    if concept_type == "object":
        for concept in previously_unlearned:
            if concept == "":
                continue
            prompt = f"An image of a {concept}"
            print(f"Adding previously unlearned object prompt: '{prompt}'")
            object_prompts.append(prompt)
        print(f"Added '{len(object_prompts)}' previously unlearned object prompts")

    # Ensure gpt doesn't generate prompts that we plan on unlearning this training run
    object_exclusions = target_concept_list if concept_type == "object" else []
    object_messages = [
        {"role": "system", "content": "You are an expert at generating creative image captions. You will generate random prompts in the format 'An image of a {object}'. Focus on concrete, recognizable objects. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line."},
        {"role": "user", "content": f'Generate {num_prompts} random prompts for images in the format "An image of a {{object}}". The objects should be diverse, concrete things like animals, vehicles, buildings, tools, etc. None of the objects should contain the words: {object_exclusions}.' if object_exclusions else f'Generate {num_prompts} random prompts for images in the format "An image of a {{object}}". The objects should be diverse, concrete things like animals, vehicles, buildings, tools, etc.'},
    ]
    if concept_type == "object" or dual_domain:
        print(f"Object GPT Query: '{object_messages[1]['content']}'")
        if dual_domain:
            num_object_prompts = (num_prompts // 2) - len(object_prompts)
        else:
            num_object_prompts = num_prompts - len(object_prompts)
        object_prompts.extend(generate_prompts_with_gpt(object_messages, num_object_prompts, object_exclusions))
    else:
        print(f"Skipping object prompt generation for concept type '{concept_type}'")
        
    # Combine all prompts
    all_prompts = style_prompts + object_prompts
    
    # Clean prompts
    all_prompts = clean_prompt(all_prompts)
    
    # Write to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for prompt in all_prompts:
            f.write(prompt + "\n")

    print(f"\nGenerated '{len(style_prompts)}' style prompts and '{len(object_prompts)}' object prompts")
    print(f"Total: '{len(all_prompts)}' prompts saved to '{file_path}'")

    return all_prompts

def clean_prompt(class_prompt_collection):
    class_prompt_collection = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', "") for x in class_prompt_collection]
    return class_prompt_collection

def generate_prompts_with_gpt(messages, target_count, exclusions):
    """Helper function to generate prompts using GPT with exclusion filtering"""
    # Store prompts
    prompt_collection = []

    # Keep track of number of queries
    numtries = 1
    
    while len(prompt_collection) < target_count and numtries <= 10:
        # Print status
        print(f"{numtries}. Querying GPT for '{target_count - len(prompt_collection)}' prompts...")
        
        try:
            # Query GPT
            response = openai.ChatCompletion.create(
                model="gpt-4.1", messages=messages
            )
            outputs = response.choices[0].message.content.split("\n")
            
            # Filter out excluded concepts and empty lines
            new_prompts = [
                x.strip() for x in outputs
                if (all(exclusion.lower() not in x.lower() for exclusion in exclusions) 
                    and x.strip() != '' 
                    and x.strip() not in prompt_collection)
            ]
            prompt_collection.extend(new_prompts)
            
            # Update conversation
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            messages.append({
                "role": "user", 
                "content": f"Generate {target_count - len(prompt_collection)} more captions in the same format"
            })
            
            # Keep only recent messages to avoid context length issues
            messages = messages[-10:]
            
        except Exception as e:
            print(f"Error querying GPT: {e}")
            break
            
        numtries += 1
    
    return prompt_collection[:target_count]
