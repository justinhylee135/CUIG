import torch


def build_orthogonal_projector(auxiliary_embeddings_matrix, device):
    """
    Construct the orthogonal projector to the orthogonal complement of the auxiliary embeddings subspace

    Args:
        auxiliary_embeddings_matrix: Auxiliary embedding matrix [embedding_dim, num_auxiliaries]
        device: Target device for computations

    Returns:
        Projection matrix P with shape [embedding_dim, embedding_dim] or None if no auxiliaries provided.
    """

    if auxiliary_embeddings_matrix is None or auxiliary_embeddings_matrix.shape[1] == 0:
        raise ValueError("ERROR: Auxiliary embedding matrix is None or has 0 auxiliary concept embedding vectors")

    # I'm switching the variable name to C to match the variable names in our paper (Section 7)
    C = auxiliary_embeddings_matrix.to(device=device, dtype=torch.float32)
    embedding_dim = C.shape[0]
    num_auxiliaries = C.shape[1]

    try:
        # Compute C^T C. Basically computes dot product of each auxiliary concept embedding with others (Gram Matrix). 
        # Captures geometry of the text embedding subspace 
        CTC = C.T @ C  # [num_auxiliaries, num_auxiliaries]

        # Regularization term is added to ensure numerical stability
        # It is scaled by the average magnitude of C^T C
        reg_strength = max(1e-4, 1e-6 * torch.trace(CTC).item() / num_auxiliaries)

        # Builds regularization matrix (Just regularization scalar times identity matrix)
        reg_term = reg_strength * torch.eye(num_auxiliaries, device=device, dtype=CTC.dtype)

        # Computing eigenvalues tells you how much independent information is in each direction
        # Also tells you whether the matrix is close to singular (really small eigenvalues is a sign of instability)
        eigenvals = torch.linalg.eigvals(CTC + reg_term).real

        # Ratio of largest eigenvalue to smallest eigenvalue gives you the condition number
        # This tells you have sensitive the inverse is to small numerical errors (Ideally close to 1)
        condition_number = eigenvals.max() / eigenvals.min()

        # If the condition number is too large, it is too unstable, so it's not worth trying to invert it
        if condition_number > 1e10:
            raise torch.linalg.LinAlgError("Matrix is ill-conditioned (condition number > 1e10)")

        # Invert regularized C^T C
        CTC_inv = torch.linalg.inv(CTC + reg_term)

        # Now we need to solve for the Orthogonal Projector onto the Orthogonal Complement of C
        # Which is: P = I - C @ CTC_inv @ C.T
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ CTC_inv @ C.T  # [embedding_dim, embedding_dim]

    except torch.linalg.LinAlgError:
        # If the matrix is too unstable we'll just use pseudo-inverse instead of actual inverse matrix
        print("WARNING: Using pseudoinverse for gradient projection")

        # Compute Moore-Penrose pseudoinverse of C^T
        C_pinv = torch.linalg.pinv(C.T)  # [num_auxiliaries, embedding_dim]

        # Solve for Orthogonal Projector onto the Orthogonal Complement of C
        # Projection matrix P = I - C @ C_pinv
        I = torch.eye(embedding_dim, device=device, dtype=C.dtype)
        P = I - C @ C_pinv # [embedding_dim, embedding_dim]

    # Return the orthogonal projector onto the orthogonal complement of the auxiliary embeddings subspace
    P = P.to(device=device, dtype=torch.float32) # [embedding_dim, embedding_dim]
    return P


def apply_gradient_projection(unet, aux_orth_proj, device, accelerator=None):
    """
    Apply gradient projection using pre-filtered embedding matrix.

    Args:
        UNet: The UNet model
        aux_orth_proj: The orthogonal projector onto the orthogonal complement of the auxiliary embedding subspace
        device: Device to run computations on
        accelerator: HuggingFace accelerator (optional)
    """
    
    # Ensure our projector is valid
    if aux_orth_proj is None: raise ValueError("Auxiliary Orthogonal Projector is None")
    
    # Ensure our projector is on the right device
    aux_orth_proj = aux_orth_proj.to(device=device)

    # Get the datatype of the projector
    dtype_of_aux_orth_proj = aux_orth_proj.dtype

    # If we're using accelerator make sure to unwrap the unet first
    if accelerator: unet = accelerator.unwrap_model(unet)

    # Iterate through all gradients for the K,V matrices in cross-attention layers
    for name, param in unet.named_parameters():
        if (param.grad is not None and # Ensure gradient exists
            'attn2' in name and # Ensure cross attention layer
            ('to_k' in name or 'to_v' in name)): # Ensure key or value matrix

            # Original shape of the gradient
            gradient_shape = param.grad.shape

            # Standard linear layer: [out_features, in_features]
            if len(gradient_shape) == 2:
                # Ensure gradient is in the same datatype as the projector
                original_gradient = param.grad.to(dtype_of_aux_orth_proj)  # [out_features, in_features]

                # Apply the orthogonal projector to the gradient
                projected_gradient = original_gradient @ aux_orth_proj  # Project along input dimension

                # Update the UNet with the new gradient
                param.grad.data = projected_gradient.to(param.grad.dtype)

            elif len(gradient_shape) == 1: # Bias terms has dimension 1
                # Skip projection for bias terms
                continue
            else:
                # Handle unexpected gradient shapes by reshaping
                reshaped_gradient = param.grad.view(gradient_shape[0], -1).to(dtype_of_aux_orth_proj)

                # Project reshaped gradient
                projected_gradient = reshaped_gradient @ aux_orth_proj

                # Return the project gradient back to the original shape
                projected_gradient = projected_gradient.view(gradient_shape).to(param.grad.dtype)
                
                # Update the UNet with the new gradient
                param.grad.data = projected_gradient