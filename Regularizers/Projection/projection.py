"""Public facade for projection regularizer utilities."""

from Regularizers.Projection.src.gradient_projection import apply_gradient_projection, build_orthogonal_projector
from Regularizers.Projection.src.auxiliary_embeddings import get_auxiliary_embeddings
from Regularizers.Projection.src.prompt_generation import generate_auxiliary_prompts

__all__ = [
    "generate_auxiliary_prompts",
    "get_auxiliary_embeddings",
    "build_orthogonal_projector",
]
