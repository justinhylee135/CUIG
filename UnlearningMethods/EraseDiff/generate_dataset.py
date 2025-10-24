import argparse
import os

import pandas as pd
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image
from tqdm.auto import tqdm


def generate_images(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
    scheduler_type="lms",
    base_model="CompVis/stable-diffusion-v1-4",
):
    """
    Function to generate images using diffusers pipeline

    The program requires the prompts to be in a csv format with headers:
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'evaluation_seed' or 'seed' (the initial seed for random number generation)

    Parameters
    ----------
    model_name : str
        name/path of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.
    scheduler_type : str, optional
        Type of scheduler to use. The default is 'lms'.
    base_model : str, optional
        Base model ID if loading from original SD. The default is 'CompVis/stable-diffusion-v1-4'.

    Returns
    -------
    None.

    """

    # CHANGE 1: Load pipeline directly instead of individual components
    print(f"Loading model: {model_name}")
    
    # Check if model_name is a path to a trained model or a HuggingFace model ID
    if os.path.exists(model_name):
        # Load from local path (our trained model)
        print(f"Loading from local path: {model_name}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    elif "SD" in model_name or model_name == base_model:
        # Load base Stable Diffusion model
        print(f"Loading base model: {base_model}")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        # Try to load as HuggingFace model ID
        try:
            print(f"Attempting to load from HuggingFace: {model_name}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except:
            # Fallback: Load base model and try to load UNet weights
            print(f"Loading base model and attempting to load UNet from: {model_name}")
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            # Try to load UNet weights if it's a .pt file
            if model_name.endswith('.pt'):
                pipe.unet.load_state_dict(torch.load(model_name))
            else:
                raise ValueError(f"Could not load model from: {model_name}")

    # CHANGE 2: Set up scheduler based on scheduler_type parameter
    if scheduler_type == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        print(f"Unknown scheduler type {scheduler_type}, using default LMS")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    # Move pipeline to device
    pipe = pipe.to(device)
    
    # CHANGE 3: Enable memory efficient attention if available (for faster generation)
    try:
        pipe.enable_attention_slicing()
        print("Enabled attention slicing for memory efficiency")
    except:
        pass

    # Load prompts from CSV
    df = pd.read_csv(prompts_path)
    
    # Handle different possible column names for seed
    if 'evaluation_seed' in df.columns:
        seed_column = 'evaluation_seed'
    elif 'seed' in df.columns:
        seed_column = 'seed'
    else:
        raise ValueError("CSV must contain either 'evaluation_seed' or 'seed' column")

    # Create output folder
    folder_path = f"{save_path}/{model_name.replace('/', '_')}"
    os.makedirs(folder_path, exist_ok=True)

    # CHANGE 4: Simplified generation loop using pipeline
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt = str(row.prompt)
        seed = int(row[seed_column])
        case_number = int(row.case_number)
        
        # Skip if before from_case
        if case_number < from_case:
            continue

        print(f"\nGenerating images for case {case_number}: {prompt}")
        
        # CHANGE 5: Generate images in batches for efficiency
        all_images = []
        
        # We'll generate in smaller batches to avoid memory issues
        batch_size = min(num_samples, 4)  # Generate max 4 images at a time
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Calculate how many images to generate in this batch
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Set up generator with seed (increment seed for each batch for variety)
            generator = torch.Generator(device=device).manual_seed(seed + batch_idx)
            
            # CHANGE 6: Use pipeline's __call__ method for generation
            images = pipe(
                prompt=[prompt] * current_batch_size,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                negative_prompt=[""] * current_batch_size,  # Empty negative prompt
                generator=generator,
            ).images
            
            all_images.extend(images)
        
        # CHANGE 7: Save images with consistent naming
        for img_idx, image in enumerate(all_images[:num_samples]):
            image.save(f"{folder_path}/{case_number}_{img_idx}.png")
            
        print(f"Saved {len(all_images[:num_samples])} images for case {case_number}")

    print(f"\nGeneration complete! Images saved to {folder_path}")


def generate_images_batch(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
    scheduler_type="lms",
    base_model="CompVis/stable-diffusion-v1-4",
    batch_size=4,
):
    """
    Alternative function that generates all samples for a prompt in a single batch.
    This can be faster but uses more memory.
    """
    
    print(f"Loading model: {model_name}")
    
    # Load pipeline (same as above)
    if os.path.exists(model_name):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    elif "SD" in model_name or model_name == base_model:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except:
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if model_name.endswith('.pt'):
                pipe.unet.load_state_dict(torch.load(model_name))
            else:
                raise ValueError(f"Could not load model from: {model_name}")

    # Set scheduler
    scheduler_map = {
        "lms": LMSDiscreteScheduler,
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
    }
    
    scheduler_class = scheduler_map.get(scheduler_type, LMSDiscreteScheduler)
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
    
    # Enable optimizations
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except:
        pass

    # Load prompts
    df = pd.read_csv(prompts_path)
    seed_column = 'evaluation_seed' if 'evaluation_seed' in df.columns else 'seed'
    
    folder_path = f"{save_path}/{model_name.replace('/', '_')}"
    os.makedirs(folder_path, exist_ok=True)

    # Generate images
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt = str(row.prompt)
        seed = int(row[seed_column])
        case_number = int(row.case_number)
        
        if case_number < from_case:
            continue

        print(f"\nGenerating {num_samples} images for case {case_number}")
        
        # CHANGE 8: Generate all samples with different seeds
        images = []
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            generators = [
                torch.Generator(device=device).manual_seed(seed + i + j) 
                for j in range(current_batch)
            ]
            
            batch_images = pipe(
                prompt=[prompt] * current_batch,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images
            
            images.extend(batch_images)
        
        # Save images
        for img_idx, image in enumerate(images[:num_samples]):
            image.save(f"{folder_path}/{case_number}_{img_idx}.png")

    print(f"\nGeneration complete! Images saved to {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", 
        description="Generate Images using Diffusers Pipeline"
    )
    
    # CHANGE 9: Updated arguments to be more flexible
    parser.add_argument(
        "--model_name", 
        help="name of model, path to model folder, or HuggingFace model ID", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--prompts_path", 
        help="path to csv file with prompts", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--save_path", 
        help="folder where to save images", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:0",
    )
    parser.add_argument(
        "--guidance_scale",
        help="guidance scale for classifier-free guidance",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--image_size",
        help="image size (height and width)",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--from_case",
        help="continue generating from case_number",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples per prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--ddim_steps",
        help="number of denoising steps",
        type=int,
        required=False,
        default=50,
    )
    # CHANGE 10: Add scheduler type argument
    parser.add_argument(
        "--scheduler",
        help="scheduler type (lms, ddim, pndm, euler, euler_a, dpm)",
        type=str,
        required=False,
        default="lms",
    )
    # CHANGE 11: Add base model argument for flexibility
    parser.add_argument(
        "--base_model",
        help="base model to use if loading custom weights",
        type=str,
        required=False,
        default="CompVis/stable-diffusion-v1-4",
    )
    # CHANGE 12: Add batch generation option
    parser.add_argument(
        "--batch_generation",
        help="use batch generation (faster but uses more memory)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size for batch generation",
        type=int,
        required=False,
        default=4,
    )
    
    args = parser.parse_args()

    # Choose generation function based on batch_generation flag
    if args.batch_generation:
        generate_images_batch(
            args.model_name,
            args.prompts_path,
            args.save_path,
            device=args.device,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            num_samples=args.num_samples,
            from_case=args.from_case,
            scheduler_type=args.scheduler,
            base_model=args.base_model,
            batch_size=args.batch_size,
        )
    else:
        generate_images(
            args.model_name,
            args.prompts_path,
            args.save_path,
            device=args.device,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            num_samples=args.num_samples,
            from_case=args.from_case,
            scheduler_type=args.scheduler,
            base_model=args.base_model,
        )