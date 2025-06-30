# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import ast

# Third Party
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

# Local
from ties import ties_merge
from uniform import uniform_merge
from task_arithmetic import task_arithmetic_merge

def main():
    parser = argparse.ArgumentParser(
        description="Merge diffusion model checkpoints using various methods"
    )
    parser.add_argument(
        "--base_model_dir", 
        type=str, 
        required=True,
        help="Path to base model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--ckpt_paths", 
        type=str, 
        required=True,
        help="List of checkpoint paths to merge"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--merge_method", 
        type=str, 
        default="ties",
        choices=["uniform", "task_arithmetic", "ties"],
        help="Merging method to use"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to use for merging"
    )
    parser.add_argument(
        "--key_filter", 
        type=str, 
        default=None,
        help="Only merge parameters whose keys contain this string (e.g., 'attn2' for cross-attention)"
    )
    
    # TIES-specific arguments
    parser.add_argument(
        "--ties_lambda", 
        type=float, 
        default=1.0,
        help="Lambda scaling factor for TIES merging"
    )
    parser.add_argument(
        "--ties_topk", 
        type=float, 
        default=0.3,
        help="Top-K fraction for TIES merging"
    )
    parser.add_argument(
        "--ties_merge_func", 
        type=str, 
        default="mean",
        choices=["mean", "sum", "max"],
        help="Merging function for TIES (mean, sum, or max)"
    )
    
    # Task arithmetic arguments
    parser.add_argument(
        "--ta_lambda", 
        type=float, 
        default=1.0,
        help="Lambda scaling factor for task arithmetic"
    )
    
    args = parser.parse_args()
    
    # Parse checkpoint list
    checkpoint_paths = ast.literal_eval(args.ckpt_paths)
    
    # Create output directory
    output_dir = Path(args.save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    print(f"Loading base model from {args.base_model_dir}...")
    base_pipeline = StableDiffusionPipeline.from_pretrained(args.base_model_dir).to(args.device)
    base_unet = base_pipeline.unet
    base_state_dict = base_unet.state_dict()
    
    # Load checkpoints
    print("Loading checkpoints...")
    ckpt_state_dicts = []
    for i, ckpt_path in enumerate(checkpoint_paths, start=1):
        if not Path(ckpt_path).exists(): raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist")
        print(f"{i}. '{ckpt_path}'")
        unet_state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        ckpt_state_dicts.append(unet_state_dict)
    print(f"Loaded {len(ckpt_state_dicts)} checkpoints to device '{args.device}'")
    
    # Perform merging
    print(f"Performing '{args.merge_method}' merging with keyfilter: {args.key_filter}")
    if args.merge_method == "uniform":
        merged_state_dict = uniform_merge(
            base_state_dict, 
            ckpt_state_dicts, 
            args.key_filter
        )
    
    elif args.merge_method == "task_arithmetic":
        merged_state_dict = task_arithmetic_merge(
            base_state_dict,
            ckpt_state_dicts,
            args.key_filter,
            args.ta_lambda
        )
    
    elif args.merge_method == "ties":
        merged_state_dict = ties_merge(
            base_state_dict,
            ckpt_state_dicts,
            args.key_filter,
            args.ties_lambda,
            args.ties_topk,
            args.ties_merge_func
        )
    
    else:
        raise ValueError(f"Unknown merge method: {args.merge_method}")
    
    # Save new merged state dictionary
    torch.save(merged_state_dict, args.save_path)
    print(f"Merged model saved to {args.save_path}")

if __name__ == "__main__":
    main()