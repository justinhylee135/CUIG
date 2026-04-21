import torch


def get_auxiliary_embeddings(auxiliary_prompts, text_encoder, tokenizer, device):
    """
    Encode auxiliary prompts into text embeddings and extract the embeddings of just the auxiliary concept (ignore fromatting prefix).

    Args:
        auxiliary_prompts (list): List of auxiliary prompts to encode.
        text_encoder: The text encoder model.
        tokenizer: The tokenizer for the text encoder.
        device: Device to run computations on (e.g., 'cuda' or 'cpu').
    """
    # This helper builds the matrix C used by gradient projection:
    #   C.shape == [embedding_dim, num_auxiliaries]
    # Each column is the extracted embedding for the auxiliary concept token(s)
    # from one prompt (e.g., stripping off prefix text like "an image of a").

    # Ensure we have at least 1 auxiliary prompt
    if len(auxiliary_prompts) <= 0: 
        raise ValueError("No auxiliary prompts provided for encoding.")
    else:
        print(f"Encoding '{len(auxiliary_prompts)}' auxiliary prompts...")

    # Store one extracted concept embedding per prompt before stacking into C.
    auxiliary_embeddings_list = []

    # This is preprocessing only; no gradients should be tracked.
    with torch.no_grad():

        # Iterate through each auxiliary prompt
        for aux_prompt in auxiliary_prompts:

            # Tokenize auxiliary prompt and move token IDs / attention mask to device.
            aux_token_ids = tokenizer(
                aux_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Encode the full prompt sequence. We will later isolate only the
            # auxiliary concept token(s) from the full sequence embedding.
            aux_emb = text_encoder(aux_token_ids.input_ids)[0]

            # Extract just the concept embedding(s), ignoring formatting prefixes.
            extracted_aux_emb = extract_aux_emb_after_prefix_format(aux_emb.unsqueeze(0), aux_token_ids, tokenizer)

            # Store on CPU to avoid accumulating GPU memory while iterating prompts.
            auxiliary_embeddings_list.append(extracted_aux_emb.cpu())

    print(f"Successfully encoded '{len(auxiliary_embeddings_list)}' auxiliary concept embeddings")

    print("Now filtering to remove redundant embeddings...")

    # Stack columns into the auxiliary embedding matrix used by projection.
    auxiliary_embeddings_matrix = torch.stack(auxiliary_embeddings_list, dim=1)

    # Remove near-duplicate columns so C is less likely to be rank-deficient.
    auxiliary_embeddings_matrix = remove_similar_embeddings_from_matrix(auxiliary_embeddings_matrix)

    print(f"Final auxiliary embedding matrix shape: '{auxiliary_embeddings_matrix.shape}'")

    return auxiliary_embeddings_matrix


def extract_aux_emb_after_prefix_format(aux_emb, aux_token_ids, tokenizer, verbose=True):
    """
    Extract auxiliary concept embeddings after known prompt prefixes.
    If the concept spans multiple tokens, average their embeddings.

    Args:
        aux_emb: Full auxiliary prompt embeddings [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim]
        aux_token_ids: Auxiliary prompt tokenized inputs
        tokenizer: The tokenizer used

    Returns:
        Tensor: Averaged embedding for concept tokens
    """
    # Step 1: normalize shape so downstream indexing is always [batch, seq, hidden].
    aux_emb = _normalize_aux_embedding_shape(aux_emb)
    aux_prompt_num_tokens = aux_emb.shape[1]

    # Convert token IDs back to token strings so we can locate the prefix format.
    aux_tokens = tokenizer.convert_ids_to_tokens(aux_token_ids.input_ids[0])

    # Step 2: detect where the auxiliary concept starts (after a known prefix).
    aux_concept_start_idx = _find_aux_concept_start_idx(aux_tokens, tokenizer)
    if aux_concept_start_idx is None:
        # If the prompt does not match any expected template, we degrade gracefully
        # by using the last meaningful token embedding instead of crashing.
        return _fallback_to_last_meaningful_token(
            aux_emb=aux_emb,
            aux_token_ids=aux_token_ids,
            aux_tokens=aux_tokens,
            aux_prompt_num_tokens=aux_prompt_num_tokens,
            warning_msg=f"Warning: Could not find expected prefix in tokens: '{aux_tokens}'",
        )

    # Step 3: determine the valid token span of the prompt (before padding).
    aux_prompt_attention_mask = aux_token_ids.attention_mask[0]
    aux_prompt_last_meaningful_idx = min(aux_prompt_attention_mask.sum().item() - 1, aux_prompt_num_tokens - 1)

    # Clamp start/end indices to tensor bounds before slicing/index generation.
    aux_concept_start_idx = max(0, min(aux_concept_start_idx, aux_prompt_num_tokens - 1))
    aux_concept_end_idx = min(aux_prompt_last_meaningful_idx + 1, aux_prompt_num_tokens)

    # Step 4: collect candidate concept indices and remove special/pad tokens.
    aux_concept_indices = _collect_valid_aux_concept_indices(
        aux_tokens=aux_tokens,
        aux_prompt_num_tokens=aux_prompt_num_tokens,
        start_idx=aux_concept_start_idx,
        end_idx=aux_concept_end_idx,
    )

    if not aux_concept_indices:
        # Prefix was found, but there were no valid concept tokens afterward.
        # Fall back to the last meaningful token for robustness.
        return _fallback_to_last_meaningful_token(
            aux_emb=aux_emb,
            aux_token_ids=aux_token_ids,
            aux_tokens=aux_tokens,
            aux_prompt_num_tokens=aux_prompt_num_tokens,
            warning_msg=f"Warning: No valid concept tokens found after prefix in: '{aux_tokens}'",
        )

    # Final defensive bounds check before tensor indexing.
    aux_concept_indices = [idx for idx in aux_concept_indices if 0 <= idx < aux_prompt_num_tokens]
    if not aux_concept_indices:
        print("Error: All concept indices out of bounds!")
        return aux_emb[0, 0, :]

    try:
        # Step 5: gather embeddings for the concept token span.
        aux_concept_only_emb = aux_emb[0, aux_concept_indices, :]

        # If auxiliary concept is made up of more than 1 token
        if len(aux_concept_indices) > 1:
            # Average the aux concept embeddings
            if verbose:
                print(
                    f"Averaged concept tokens: "
                    f"{[aux_tokens[i] for i in aux_concept_indices]} ({len(aux_concept_indices)} tokens)"
                )
            return aux_concept_only_emb.mean(dim=0)
        
        # If auxiliary concept is just 1 token than just extact it and then return
        if verbose:
            print(
                f"Extracted concept tokens: "
                f"{[aux_tokens[i] for i in aux_concept_indices]} ({len(aux_concept_indices)} tokens)"
            )
        return aux_concept_only_emb[0]

    except Exception as e:
        # Defensive fallback to avoid breaking training if token extraction hits an
        # unexpected shape/indexing issue on a rare prompt.
        print(f"Error extracting concept embeddings: {e}")
        print(f"  aux_tokens: '{aux_tokens}'")
        print(f"  aux_concept_indices: '{aux_concept_indices}'")
        print(f"  aux_emb.shape: '{aux_emb.shape}'")
        return aux_emb[0, 0, :]


def _normalize_aux_embedding_shape(aux_emb):
    """Normalize auxiliary embedding tensor to [batch, seq_len, hidden_dim]."""
    # The caller can pass embeddings in slightly different shapes depending on
    # whether they came from a text encoder directly or from a pipeline helper.
    # We normalize here so the extraction code only needs to handle one shape.
    if len(aux_emb.shape) == 2:
        return aux_emb.unsqueeze(0)
    if len(aux_emb.shape) == 4:
        return aux_emb.squeeze(1)
    if len(aux_emb.shape) == 3:
        return aux_emb
    raise ValueError(f"Unexpected aux_emb shape: {aux_emb.shape}")


def _get_last_meaningful_token_idx(aux_token_ids, aux_prompt_num_tokens):
    # We use attention_mask to find the real (unpadded) prompt length.
    # `-2` intentionally skips the final special/end token in most CLIP tokenizers.
    aux_prompt_attention_mask = aux_token_ids.attention_mask[0]
    last_token_idx = min(aux_prompt_attention_mask.sum().item() - 2, aux_prompt_num_tokens - 1)
    return max(0, last_token_idx)


def _fallback_to_last_meaningful_token(aux_emb, aux_token_ids, aux_tokens, aux_prompt_num_tokens, warning_msg):
    # Centralized fallback path used when prefix parsing fails or yields no valid
    # concept tokens. Keeping this in one helper avoids duplicated debug prints.
    aux_prompt_attention_mask = aux_token_ids.attention_mask[0]
    print(
        f"Auxiliary prompt number of tokens: '{aux_prompt_num_tokens}', "
        f"Attention Mask Length: '{aux_prompt_attention_mask.sum().item()}'"
    )
    aux_prompt_last_token_idx = _get_last_meaningful_token_idx(aux_token_ids, aux_prompt_num_tokens)
    print(warning_msg)
    print(
        f"Using last meaningful token '{aux_tokens[aux_prompt_last_token_idx]}' "
        f"at index '{aux_prompt_last_token_idx}' as fallback"
    )
    return aux_emb[0, aux_prompt_last_token_idx, :]


def _get_aux_prefix_token_variants(tokenizer):
    # Supported prompt formats for extracting the concept region.
    # If new prompt templates are introduced, add their tokenized prefixes here.
    return [
        tokenizer.tokenize("an image in the style of"),
        tokenizer.tokenize("an image of a"),
        tokenizer.tokenize("an image of"),  # Without "a"
    ]


def _find_aux_concept_start_idx(aux_tokens, tokenizer):
    # Return the index of the first concept token (immediately after the prefix).
    # We try known formats in order and stop at the first match.
    for prefix_tokens in _get_aux_prefix_token_variants(tokenizer):
        prefix_start_idx = find_prefix_format_starting_index(aux_tokens, prefix_tokens)
        if prefix_start_idx is not None:
            return prefix_start_idx + len(prefix_tokens)
    return None


def _is_special_or_padding_token(token):
    # Filter out tokenizer special tokens and padding when extracting concept tokens.
    return token.startswith('[') or token in {'<|endoftext|>', '<pad>'}


def _collect_valid_aux_concept_indices(aux_tokens, aux_prompt_num_tokens, start_idx, end_idx):
    # Build the candidate concept token span and filter it to valid, non-special tokens.
    valid_indices = []
    for idx in range(start_idx, end_idx):
        if 0 <= idx < aux_prompt_num_tokens and idx < len(aux_tokens):
            if not _is_special_or_padding_token(aux_tokens[idx]):
                valid_indices.append(idx)
    return valid_indices

def _normalize_aux_embedding_shape(aux_emb):
    """Normalize auxiliary embedding tensor to [batch, seq_len, hidden_dim]."""
    # The caller can pass embeddings in slightly different shapes depending on
    # whether they came from a text encoder directly or from a pipeline helper.
    # We normalize here so the extraction code only needs to handle one shape.
    if len(aux_emb.shape) == 2:
        return aux_emb.unsqueeze(0)
    if len(aux_emb.shape) == 4:
        return aux_emb.squeeze(1)
    if len(aux_emb.shape) == 3:
        return aux_emb
    raise ValueError(f"Unexpected aux_emb shape: {aux_emb.shape}")


def _get_last_meaningful_token_idx(aux_token_ids, aux_prompt_num_tokens):
    # We use attention_mask to find the real (unpadded) prompt length.
    # `-2` intentionally skips the final special/end token in most CLIP tokenizers.
    aux_prompt_attention_mask = aux_token_ids.attention_mask[0]
    last_token_idx = min(aux_prompt_attention_mask.sum().item() - 2, aux_prompt_num_tokens - 1)
    return max(0, last_token_idx)


def _fallback_to_last_meaningful_token(aux_emb, aux_token_ids, aux_tokens, aux_prompt_num_tokens, warning_msg):
    # Centralized fallback path used when prefix parsing fails or yields no valid
    # concept tokens. Keeping this in one helper avoids duplicated debug prints.
    aux_prompt_attention_mask = aux_token_ids.attention_mask[0]
    print(
        f"Auxiliary prompt number of tokens: '{aux_prompt_num_tokens}', "
        f"Attention Mask Length: '{aux_prompt_attention_mask.sum().item()}'"
    )
    aux_prompt_last_token_idx = _get_last_meaningful_token_idx(aux_token_ids, aux_prompt_num_tokens)
    print(warning_msg)
    print(
        f"Using last meaningful token '{aux_tokens[aux_prompt_last_token_idx]}' "
        f"at index '{aux_prompt_last_token_idx}' as fallback"
    )
    return aux_emb[0, aux_prompt_last_token_idx, :]


def _get_aux_prefix_token_variants(tokenizer):
    # Supported prompt formats for extracting the concept region.
    # If new prompt templates are introduced, add their tokenized prefixes here.
    return [
        tokenizer.tokenize("an image in the style of"),
        tokenizer.tokenize("an image of a"),
        tokenizer.tokenize("an image of"),  # Without "a"
    ]


def _find_aux_concept_start_idx(aux_tokens, tokenizer):
    # Return the index of the first concept token (immediately after the prefix).
    # We try known formats in order and stop at the first match.
    for prefix_tokens in _get_aux_prefix_token_variants(tokenizer):
        prefix_start_idx = find_prefix_format_starting_index(aux_tokens, prefix_tokens)
        if prefix_start_idx is not None:
            return prefix_start_idx + len(prefix_tokens)
    return None


def _is_special_or_padding_token(token):
    # Filter out tokenizer special tokens and padding when extracting concept tokens.
    return token.startswith('[') or token in {'<|endoftext|>', '<pad>'}


def _collect_valid_aux_concept_indices(aux_tokens, aux_prompt_num_tokens, start_idx, end_idx):
    # Build the candidate concept token span and filter it to valid, non-special tokens.
    valid_indices = []
    for idx in range(start_idx, end_idx):
        if 0 <= idx < aux_prompt_num_tokens and idx < len(aux_tokens):
            if not _is_special_or_padding_token(aux_tokens[idx]):
                valid_indices.append(idx)
    return valid_indices

def find_prefix_format_starting_index(aux_tokens, prefix_format_tokens):
    """
    Find the starting index of a token sequence in the full token list.
    Returns None if not found.
    """
    # Safety checks make failure modes explicit and keep the actual matching loop small.
    if len(prefix_format_tokens) == 0:
        raise ValueError(f"Prefix format tokens is length 0. Pass a valid prefix format.")
    elif len(prefix_format_tokens) > len(aux_tokens):
        print(f"Warning: prefix_format_tokens: '{prefix_format_tokens}' is length '{len(prefix_format_tokens)}'." 
              f"This is longer than aux_tokens '{aux_tokens}' which is length '{len(aux_tokens)}'. "
              f"Prefix format cannot be found in auxiliary prompt.")
        return None

    # Normalize tokens for comparison (handle BPE markers, case, etc.)
    # This keeps matching robust across tokenizer formatting conventions.
    aux_tokens_cleaned = [t.lower().replace('##', '').replace('▁', '') for t in aux_tokens]
    prefix_format_tokens_cleaned = [t.lower().replace('##', '').replace('▁', '') for t in prefix_format_tokens]

    # Search for an exact subsequence match of the cleaned prefix tokens.
    prefix_len = len(prefix_format_tokens_cleaned)
    for i in range(len(aux_tokens_cleaned) - prefix_len + 1):
        aux_tokens_slice = aux_tokens_cleaned[i : i + prefix_len]
        if aux_tokens_slice == prefix_format_tokens_cleaned:
            return i

    return None


def remove_similar_embeddings_from_matrix(auxiliary_embeddings_matrix, similarity_threshold=0.95):
    """
    Remove embeddings that are too similar to avoid rank deficiency.

    Args:
        auxiliary_embeddings_matrix: Embedding matrix [embedding_dim, num_auxiliaries]
        similarity_threshold: Remove embeddings with cosine similarity > threshold

    Returns:
        filtered_auxiliary_embeddings_matrix: Matrix with similar embeddings removed
    """

    # Trivial case: nothing to filter if there are 0 or 1 columns.
    if auxiliary_embeddings_matrix.shape[1] <= 1:
        return auxiliary_embeddings_matrix

    # Normalize embeddings for cosine similarity
    auxiliary_embeddings_matrix_normalized = auxiliary_embeddings_matrix / (auxiliary_embeddings_matrix.norm(dim=0, keepdim=True) + 1e-8)

    # Compute pairwise cosine similarities
    similarities = auxiliary_embeddings_matrix_normalized.T @ auxiliary_embeddings_matrix_normalized  # [num_auxiliary_concepts, num_auxiliary_concepts]

    # Greedy filtering: keep the first embedding, then keep a new one only if it is not too similar to any embedding already kept.
    num_auxiliaries = auxiliary_embeddings_matrix.shape[1]

    # The auxiliary concept embeddings to keep
    keep_indices = [0]  # Always keep the first one

    # Iterate through each auxiliary concept embedding
    for i in range(1, num_auxiliaries):
        # Check if this auxiliary embedding is too similar to any we're keeping
        max_similarity = similarities[i, keep_indices].max().item()

        # If it's under the threshold then keep it
        if max_similarity < similarity_threshold:
            keep_indices.append(i)

    # Return filtered auxiliary embeddings matrix
    print(f"Keeping '{len(keep_indices)}/{num_auxiliaries}' auxiliary embeddings after similarity filtering with threshold '{similarity_threshold}'")
    return auxiliary_embeddings_matrix[:, keep_indices]

def get_auxiliary_embeddings_from_diffusion_pipeline(auxiliary_prompts, diffusion_pipeline, device):
    """
    SDXL-friendly helper that leverages diffusion_pipeline.encode_prompt to build auxiliary embeddings that
    match the conditioning dimension seen by the UNet (e.g., 2048 for SDXL). Keeps
    existing get_auxiliary_embeddings intact for backward compatibility.
    """   
    # This helper builds the matrix C used by gradient projection:
    #   C.shape == [embedding_dim, num_auxiliaries]
    # Each column is the extracted embedding for the auxiliary concept token(s)
    # from one prompt (e.g., stripping off prefix text like "an image of a").

    # Check for valid number of prompts
    if len(auxiliary_prompts) <= 0: 
        raise ValueError("No auxiliary prompts provided for encoding.")
    else:
        print(f"Encoding '{len(auxiliary_prompts)}' auxiliary prompts via diffusion_pipeline.encode_prompt...")

    # Check our diffusion pipeline has an encode prompt function
    if not hasattr(diffusion_pipeline, "encode_prompt"):
        raise AttributeError("Pipeline is missing encode_prompt; cannot build auxiliary embeddings with diffusion_pipeline.")

    # Store one extracted concept embedding per prompt before stacking into C.
    auxiliary_embeddings_list = []

    # This is preprocessing only; no gradients should be tracked.
    with torch.no_grad():

        # Iterate through each auxiliary prompt
        for aux_prompt in auxiliary_prompts:
            # Tokenize auxiliary prompt and move token IDs / attention mask to device.
            aux_token_ids = diffusion_pipeline.tokenizer(
                aux_prompt,
                padding="max_length",
                max_length=diffusion_pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Get auxiliary prompt embeddings
            aux_emb, _, _, _ = diffusion_pipeline.encode_prompt(
                prompt=aux_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
            )

            # Extract just the concept embedding(s), ignoring formatting prefixes.
            extracted_aux_emb = extract_aux_emb_after_prefix_format(
                aux_emb.unsqueeze(0), aux_token_ids, diffusion_pipeline.tokenizer
            )

            # Store on CPU to avoid accumulating GPU memory while iterating prompts.
            auxiliary_embeddings_list.append(extracted_aux_emb.cpu())

    print(f"Successfully encoded '{len(auxiliary_embeddings_list)}' auxiliary embeddings via diffusion_pipeline.encode_prompt().")

    # Stack columns into the auxiliary embedding matrix used by projection.
    auxiliary_embeddings_matrix = torch.stack(auxiliary_embeddings_list, dim=1)

    # Remove near-duplicate columns so C is less likely to be rank-deficient.
    auxiliary_embeddings_matrix = remove_similar_embeddings_from_matrix(auxiliary_embeddings_matrix)

    # Return filtered auxiliary embeddings matrix
    print(f"Final auxiliary embedding matrix shape (via diffusion_pipeline): '{auxiliary_embeddings_matrix.shape}'")
    return auxiliary_embeddings_matrix
