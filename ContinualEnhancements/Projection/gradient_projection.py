import os
import torch
import torch.nn.functional as F

def apply_gradient_projection(model, filtered_embedding_matrix, device, accelerator=None):
    """
    Apply gradient projection using pre-filtered embedding matrix.
    
    Args:
        model: The UNet model
        filtered_embedding_matrix: Pre-computed and filtered embedding matrix [embedding_dim, num_anchors]
        device: Device to run computations on
        accelerator: HuggingFace accelerator (optional)
    """
    # Check if we have valid embeddings
    if filtered_embedding_matrix is None or filtered_embedding_matrix.shape[1] == 0:
        return  # No projection needed if no anchor embeddings
    
    C = filtered_embedding_matrix.to(device)  # [embedding_dim, num_anchors]
    embedding_dim = C.shape[0]
    num_anchors = C.shape[1]
    
    # Compute projection matrix: P = I - C(C^T C)^(-1)C^T
    try:
        # Normal inverse with regularization
        CTC = C.T @ C  # [num_anchors, num_anchors]
        reg_strength = max(1e-4, 1e-6 * torch.trace(CTC).item() / num_anchors)
        reg_term = reg_strength * torch.eye(num_anchors, device=device, dtype=CTC.dtype)
        
        # Check condition number for numerical stability
        eigenvals = torch.linalg.eigvals(CTC + reg_term).real
        condition_number = eigenvals.max() / eigenvals.min()
        
        if condition_number > 1e10:  # Matrix is too ill-conditioned
            raise torch.linalg.LinAlgError("Matrix is ill-conditioned")
            
        CTC_inv = torch.linalg.inv(CTC + reg_term)
        
        # Projection matrix P = I - C @ CTC_inv @ C.T
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ CTC_inv @ C.T  # [embedding_dim, embedding_dim]
        
    except torch.linalg.LinAlgError:
        # Fallback to pseudoinverse if matrix is singular
        print(f"Warning: Using pseudoinverse for gradient projection")
        C_pinv = torch.linalg.pinv(C.T)  # [num_anchors, embedding_dim]
        
        # Projection matrix P = I - C @ C_pinv
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ C_pinv
    
    # Apply projection to cross-attention gradients
    model_to_process = accelerator.unwrap_model(model) if accelerator else model
    
    for name, param in model_to_process.named_parameters():
        if (param.grad is not None and 
            'attn2' in name and 
            ('to_k' in name or 'to_v' in name)):
            
            original_shape = param.grad.shape
            
            if len(original_shape) == 2:
                # Standard linear layer: [out_features, in_features]
                grad_matrix = param.grad  # [out_features, in_features]
                projected_grad = grad_matrix @ P  # Project along input dimension
                param.grad.data = projected_grad
            elif len(original_shape) == 1:
                # Skip bias terms - no projection needed
                continue
            else:
                # Handle unexpected shapes by reshaping
                grad_reshaped = param.grad.view(original_shape[0], -1)
                projected_grad = grad_reshaped @ P
                param.grad.data = projected_grad.view(original_shape)

def get_anchor_embeddings(anchor_prompts, text_encoder, tokenizer, device):
    """    Encode anchor prompts into text embeddings.
    Args:
        anchor_prompts (list): List of anchor prompts to encode.
        text_encoder: The text encoder model.
        tokenizer: The tokenizer for the text encoder.
        device: Device to run computations on (e.g., 'cuda' or 'cpu').
    """    
    # Encode anchor prompts to get text embeddings
    if len(anchor_prompts) > 0:
        print(f"Encoding {len(anchor_prompts)} anchor prompts...")
        anchor_embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(anchor_prompts), batch_size):
            batch_prompts = anchor_prompts[i:i+batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                text_inputs = tokenizer(
                    batch_prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                # Get text embeddings
                text_embeddings = text_encoder(text_inputs.input_ids)[0]
                
                # Get last meaningful token for each prompt
                attention_mask = text_inputs.attention_mask
                last_token_indices = attention_mask.sum(dim=1) - 1
                
                # Extract embeddings for last tokens
                batch_embeddings = text_embeddings[torch.arange(len(batch_prompts)), last_token_indices, :]
                anchor_embeddings.extend([emb.cpu() for emb in batch_embeddings])
        
        print(f"Successfully encoded {len(anchor_embeddings)} anchor embeddings")
        
        print(f"Now filtering to remove redundant embeddings...")
        # Convert to tensor matrix
        anchor_embedding_tensors = []
        for emb in anchor_embeddings:
            if isinstance(emb, torch.Tensor):
                emb_tensor = emb.to(device).flatten()
            else:
                emb_tensor = torch.tensor(emb, device=device).flatten()
            anchor_embedding_tensors.append(emb_tensor)
        
        # Stack into matrix [embedding_dim, num_anchors]
        C = torch.stack(anchor_embedding_tensors, dim=1)
        
        # Remove similar embeddings once
        C_filtered = remove_similar_embeddings(C)
        
        print(f"Final anchor embedding matrix shape: {C_filtered.shape}")
        
        return C_filtered
    else:
        raise ValueError("No anchor prompts provided for encoding.")

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