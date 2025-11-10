#!/usr/bin/env python3
"""
Utilities for constructing text-embedding “vocabularies” and performing token searches using
the Stable Diffusion text encoder (diffusers implementation).

This module intentionally mirrors the public API that `train_age.py` expects:
    - save_embedding_matrix
    - search_closest_tokens
    - learn_k_means_from_input_embedding
Additional helper utilities can be invoked via the CLI for ad-hoc embedding dumps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ENGLISH_VOCAB_SIZE = 3000
IMAGENET_VOCAB_SIZE = 1000
CLIP_CHUNK = 5000
DATA_DIR = Path(__file__).resolve().parent / "data"
ENGLISH_CSV = DATA_DIR / "english_3000.csv"


def _load_english_tokens() -> Dict[str, int]:
    df = pd.read_csv(ENGLISH_CSV)
    vocab = {row["word"]: idx for idx, row in df.iterrows()}
    if len(vocab) != ENGLISH_VOCAB_SIZE:
        raise ValueError(f"Expected {ENGLISH_VOCAB_SIZE} English tokens, found {len(vocab)}")
    return vocab


def _load_imagenet_tokens() -> Dict[str, int]:
    from imagenet_labels import IMAGENET_1K  # Lazy import to avoid dependency when unused

    vocab = {}
    for index in IMAGENET_1K:
        base_token = IMAGENET_1K[index]
        token = base_token
        suffix = 1
        while token in vocab:
            suffix += 1
            token = f"{base_token} ({suffix})"
        vocab[token] = index
    if len(vocab) != IMAGENET_VOCAB_SIZE:
        raise ValueError(f"Expected {IMAGENET_VOCAB_SIZE} Imagenet tokens, found {len(vocab)}")
    return vocab


def get_vocab(model, vocab: str = "EN3K") -> Dict[str, int]:
    if vocab == "CLIP":
        return model.tokenizer.get_vocab()
    if vocab == "EN3K":
        return _load_english_tokens()
    if vocab == "Imagenet":
        return _load_imagenet_tokens()
    if vocab == "artistic":
        return
    raise ValueError(f"Unsupported vocabulary '{vocab}' (expected 'CLIP', 'EN3K', or 'Imagenet').")


def _embed_tokens(model, tokens: Sequence[str], remove_end_token: bool = False) -> torch.Tensor:
    embeddings = []
    for token in tokens:
        token_text = token.replace("</w>", "") if remove_end_token else token
        emb = model.get_learned_conditioning([token_text])
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def save_embedding_matrix(
    model,
    model_name: str = "sd-v1-5",
    save_mode: str = "array",
    vocab: str = "EN3K",
    output_dir: Path | str = "models",
):
    """
    Persist token embeddings to disk for later reuse.

    Parameters
    ----------
    model : StableDiffusionWrapper-like object
        Must expose `get_learned_conditioning` and `.tokenizer`.
    save_mode : str
        Either "array" (stacked tensor) or "dict" (token -> embedding).
    vocab : str
        One of {"CLIP", "EN3K", "Imagenet"}.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if vocab == "CLIP":
        tokenizer_vocab = get_vocab(model, vocab="CLIP")
        tokens = list(tokenizer_vocab.keys())
        for start in range(0, len(tokens), CLIP_CHUNK):
            chunk_tokens = tokens[start : start + CLIP_CHUNK]
            embeddings = _embed_tokens(model, chunk_tokens)
            if save_mode == "array":
                payload = embeddings
            elif save_mode == "dict":
                payload = {token: emb for token, emb in zip(chunk_tokens, embeddings)}
            else:
                raise ValueError("save_mode must be 'array' or 'dict'")
            target = output_dir / f"embedding_matrix_{start}_{start + len(chunk_tokens)}_{save_mode}.pt"
            torch.save(payload, target)
            print(f"[embedding_matrix] Saved {len(chunk_tokens)} tokens to {target}")
        return

    if vocab == "EN3K":
        tokens = list(get_vocab(model, "EN3K").keys())
    elif vocab == "Imagenet":
        tokens = list(get_vocab(model, "Imagenet").keys())
    else:
        raise ValueError(f"Unsupported vocab '{vocab}'")

    embeddings = _embed_tokens(model, tokens, remove_end_token=True)
    if save_mode == "array":
        payload = embeddings
    elif save_mode == "dict":
        payload = {token: emb for token, emb in zip(tokens, embeddings)}
    else:
        raise ValueError("save_mode must be 'array' or 'dict'")

    target = output_dir / f"embedding_matrix_{save_mode}_{vocab}.pt"
    torch.save(payload, target)
    print(f"[embedding_matrix] Saved {len(tokens)} tokens to {target}")


def detect_special_tokens(text: str) -> bool:
    text = text.lower()
    for char in text:
        if char not in "abcdefghijklmnopqrstuvwxyz</> ":
            return True
    return False


@torch.no_grad()
def search_closest_tokens(
    concept: str,
    model,
    k: int = 5,
    reshape: bool = True,
    sim: str = "cosine",
    model_name: str = "SD-v1-4",
    ignore_special_tokens: bool = True,
    vocab: str = "EN3K",
    foundInConceptDict: bool = False,
) -> tuple[List[str], Dict[str, torch.Tensor]]:
    """Return the top-k vocabulary tokens closest to `concept` in embedding space."""
    if foundInConceptDict:
        vocab_tokens = vocab
    else:
        vocab_tokens = get_vocab(model, vocab=vocab)
    inverse_vocab = {idx: token for token, idx in vocab_tokens.items()}
    concept_embedding = model.get_learned_conditioning([concept])
    concept_embedding = concept_embedding.flatten(start_dim=1)

    similarities = {}
    for idx in vocab_tokens.values():
        token = inverse_vocab[idx]
        if ignore_special_tokens and detect_special_tokens(token):
            continue
        token_embedding = model.get_learned_conditioning([token]).flatten(start_dim=1)
        if reshape:
            token_embedding = token_embedding.reshape(concept_embedding.shape)
        if sim == "cosine":
            similarity = F.cosine_similarity(token_embedding, concept_embedding, dim=-1)
        elif sim == "l2":
            similarity = -F.pairwise_distance(token_embedding, concept_embedding, p=2, keepdim=True)
        else:
            raise ValueError("sim must be 'cosine' or 'l2'")
        similarities[token] = similarity.squeeze().detach().cpu()

    sorted_sim = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    top_k = list(sorted_sim.keys())[:k]
    print(f"[search_closest_tokens] Top-{k} tokens for '{concept}': {top_k}")
    return top_k, sorted_sim


def my_kmean(sorted_sim_dict: Dict[str, torch.Tensor], num_centers: int, compute_mode: str = "numpy"):
    if compute_mode not in {"numpy", "torch"}:
        raise ValueError("compute_mode must be 'numpy' or 'torch'")

    values = torch.stack(
        [torch.atleast_1d(v).reshape(-1)[0].to(torch.float32) for v in sorted_sim_dict.values()]
    )
    tokens = list(sorted_sim_dict.keys())

    if compute_mode == "numpy":
        values_np = values.float().cpu().numpy().reshape(-1, 1)
        from sklearn.cluster import KMeans

        cluster_ids = KMeans(n_clusters=num_centers, random_state=0).fit(values_np).labels_
    else:
        # simple torch-based clustering (k-means++ style would be overkill here)
        values = values.float().unsqueeze(-1)  # (N, 1) in fp32 for stable distance computation
        rand_idx = torch.randperm(values.size(0))[:num_centers]
        centers = values[rand_idx]
        for _ in range(20):
            distances = torch.cdist(values, centers)
            assignments = distances.argmin(dim=1)
            for cid in range(num_centers):
                mask = assignments == cid
                if mask.any():
                    centers[cid] = values[mask].mean(dim=0)
        cluster_ids = assignments.cpu().numpy()

    cluster_dict = {}
    for token, cid in zip(tokens, cluster_ids):
        if cid not in cluster_dict:
            cluster_dict[cid] = (token, sorted_sim_dict[token], cid)
        else:
            _, ref_sim, _ = cluster_dict[cid]
            current_sim = sorted_sim_dict[token]
            if current_sim > ref_sim:
                cluster_dict[cid] = (token, current_sim, cid)
    return cluster_dict


def learn_k_means_from_input_embedding(sim_dict: Dict[str, torch.Tensor], num_centers: int = 5, compute_mode: str = "numpy"):
    """
    Given the similarity dictionary returned by `search_closest_tokens`, cluster tokens and
    return a representative subset.
    """
    if num_centers <= 0 or len(sim_dict) <= num_centers:
        return list(sim_dict.keys())
    centers = my_kmean(sim_dict, num_centers, compute_mode)
    return list(token for token, _, _ in centers.values())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or inspect SD text embeddings via diffusers.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Diffusers model id or path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-embeddings", action="store_true", help="Dump embedding matrices to disk.")
    parser.add_argument("--vocab", default="EN3K", choices=["CLIP", "EN3K", "Imagenet"])
    parser.add_argument("--save-mode", default="dict", choices=["dict", "array"])
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--concept", type=str, help="Optional concept for interactive closest-token search.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" and torch.cuda.is_available() else torch.float32

    from diffusers_utils import load_diffusers_model

    model = load_diffusers_model(args.model, device, dtype)

    if args.save_embeddings:
        save_embedding_matrix(model, model_name=Path(args.model).name, save_mode=args.save_mode, vocab=args.vocab, output_dir=args.output_dir)

    if args.concept:
        search_closest_tokens(args.concept, model, k=args.top_k, vocab=args.vocab)


if __name__ == "__main__":
    main()
