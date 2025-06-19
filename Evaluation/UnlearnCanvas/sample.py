# Standard Library
import argparse
import json
import os
import ast
from PIL import Image

# Third Party
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Sampler',
        description='Sample Images from Unlearned Model')
    # Model parameters
    parser.add_argument('--unet_ckpt_path', help='Path to UNet ckpt', type=str, required=True)
    parser.add_argument('--pipeline_dir', help='Directory for Diffusers pipeline', type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Generation parameters
    parser.add_argument('--output_dir', help='Folder to save images', type=str, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[188, 288, 588, 688, 888])
    parser.add_argument('--cfg_txt', help='Guidance strength', type=float, required=False, default=9.0)
    parser.add_argument('--steps', help='Inference steps', type=int, required=False, default=100)
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space", )

    # Concept Subset parameters
    parser.add_argument("--themes_subset", type=str, default=None)
    parser.add_argument("--objects_subset", type=str, default=None)

    args = parser.parse_args()

    # Constants
    theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
                    "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
                    "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
                    "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
                    "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
                    "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
                    "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
                    "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]
    class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                    "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                    "Trees", "Waterfalls"]


    # Set subset of concepts to generate
    themes_subset = theme_available
    objects_subset = class_available
    if args.themes_subset is not None:
        themes_subset = ast.literal_eval(args.themes_subset)
    if args.objects_subset is not None:
        objects_subset = ast.literal_eval(args.objects_subset)

    # Set the cuda visible device to the specified device
    torch.cuda.set_device(args.device)
    device = "cuda"

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(args.pipeline_dir, torch_dtype=torch.float16, use_safetensors=True, allow_pickle=True).to("cuda")
    if ".safetensors" in args.unet_ckpt_path:
        unet_state_dict = load_file(args.unet_ckpt_path, device="cuda")
    else:
        unet_state_dict = torch.load(args.unet_ckpt_path, map_location="cuda")

    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded UNet from {args.unet_ckpt_path}")
    print(f"Loaded keys: {len(unet_state_dict)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    # Disable NSFW checker (sometimes incorrectly flags images)
    def dummy(images, **kwargs):
            return images, [False]
    pipe.safety_checker = dummy

    # Define output directory for images
    output_dir = args.output_dir
    print(f"Saving generated images to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Start generating images
    total_iterations = len(args.seed) * len(themes_subset) * len(objects_subset)
    with tqdm(total=total_iterations, desc="Generating images", unit="image") as pbar:
        for seed in args.seed:
            seed_everything(seed) # Set seed for reproducibility
            for theme in themes_subset:
                for obj in objects_subset:
                    # Set output path for image
                    output_path = os.path.join(args.output_dir, f"{theme}_{obj}_seed{seed}.jpg")
                    
                    # Skip if image already exists
                    if os.path.exists(output_path):
                        print(f"Image already exists! Skipping: {output_path}!")
                        pbar.update(1)
                        continue

                    # Set prompt
                    prompt = f"A {obj} image in {theme} style"

                    # Generate image
                    image = pipe(prompt=prompt, width=args.W, height=args.H, num_inference_steps=args.steps, guidance_scale=args.cfg_txt).images[0]
                    
                    # Save image
                    image.save(output_path)
                    pbar.update(1)