import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from einops import rearrange

# CHANGE 1: Import diffusers components instead of CompVis
from diffusers import (
    StableDiffusionPipeline
)

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def read_text_lines(path):
    """Read text file and return list of lines"""
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


class CenterSquareCrop:
    """Center crop to square"""
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        return F.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    """Get image transformation pipeline"""
    transform = torch_transforms.Compose([
        CenterSquareCrop(),
        torch_transforms.Resize(size, interpolation=interpolation),
    ])
    return transform


# CHANGE 2: Replace setup_model with diffusers version
def setup_diffusers_model(model_id, device, dtype=torch.float32):
    """
    Load a diffusers model from HuggingFace hub or local path
    
    Args:
        model_id: HuggingFace model ID or local path to diffusers model
        device: Device to load model on
        dtype: Data type for model (float32 or float16)
    
    Returns:
        Tuple of (unet, vae, text_encoder, tokenizer, scheduler)
    """
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Extract components
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    unet = pipe.unet.to(device)
    scheduler = pipe.scheduler
    
    # Set to eval mode (we'll only train UNet typically)
    vae.eval()
    text_encoder.eval()
    
    return unet, vae, text_encoder, tokenizer, scheduler


# CHANGE 3: Add alternative setup for loading from checkpoint file
def setup_model_from_checkpoint(ckpt_path, model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
    """
    Load a model from a checkpoint file (for compatibility with old scripts)
    
    Args:
        ckpt_path: Path to checkpoint file (.pt or .ckpt)
        model_id: Base model ID to load architecture from
        device: Device to load model on
    
    Returns:
        Tuple of (unet, vae, text_encoder, tokenizer, scheduler)
    """
    # First load the base model
    unet, vae, text_encoder, tokenizer, scheduler = setup_diffusers_model(model_id, device)
    
    # Load checkpoint
    if ckpt_path.endswith('.ckpt') or ckpt_path.endswith('.pt'):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Try to load UNet weights
        # This assumes the checkpoint contains UNet weights in a compatible format
        try:
            unet.load_state_dict(state_dict, strict=False)
            print(f"Loaded UNet weights from {ckpt_path}")
        except:
            print(f"Warning: Could not load weights from {ckpt_path}, using base model")
    
    return unet, vae, text_encoder, tokenizer, scheduler


# CHANGE 4: Update StyleDataset to support diffusers preprocessing
class StyleDataset(Dataset):
    """
    Dataset for style transfer/removal tasks
    
    Supports both [-1, 1] normalization (for diffusers) and [0, 1] normalization
    """
    def __init__(self, data_path_list, prompt_path_list, transform=None, normalize_to_neg_one=True):
        """
        Args:
            data_path_list: Path to file containing image paths
            prompt_path_list: Path to file containing prompts
            transform: Torchvision transforms to apply
            normalize_to_neg_one: If True, normalize to [-1, 1] (diffusers), 
                                 else [0, 1]
        """
        self.image_paths = []
        for dpath in data_path_list:
            img_path_list = read_text_lines(dpath)
            self.image_paths.extend(img_path_list)
            print(f"Loaded {len(img_path_list)} images from {dpath}")
            
        self.prompts = []
        for ppath in prompt_path_list:
            prompt_list = read_text_lines(ppath)
            self.prompts.extend(prompt_list)
            print(f"Loaded {len(prompt_list)} prompts from {ppath}")

        assert len(self.image_paths) == len(self.prompts), \
            f"images.txt ({len(self.image_paths)}) and prompts.txt ({len(self.prompts)}) must match"

        self.transform = transform
        self.normalize_to_neg_one = normalize_to_neg_one

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        prompt = self.prompts[idx]

        # Apply transforms (resize, crop, etc.)
        if self.transform:
            image = self.transform(image)

        # Convert to tensor and normalize
        image = torch.tensor(np.array(image)).float() / 255.0
        
        # CHANGE 5: Support both normalization ranges
        if self.normalize_to_neg_one:
            # Normalize to [-1, 1] for diffusers
            image = 2 * image - 1
        # else image stays in [0, 1]
        
        # Rearrange to CHW format
        image = rearrange(image, "h w c -> c h w")

        return image, prompt


# CHANGE 6: Add a dataset for working with image-caption pairs (common in diffusers)
class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs, commonly used with diffusers
    """
    def __init__(self, images_dir, captions_file=None, transform=None, normalize_to_neg_one=True):
        """
        Args:
            images_dir: Directory containing images
            captions_file: Optional file with captions (one per line, matching image order)
            transform: Torchvision transforms to apply
            normalize_to_neg_one: If True, normalize to [-1, 1]
        """
        self.images_dir = Path(images_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                 list(self.images_dir.glob("*.png")))
        
        if captions_file:
            self.captions = read_text_lines(captions_file)
            assert len(self.captions) == len(self.image_files), \
                f"Number of captions ({len(self.captions)}) must match number of images ({len(self.image_files)})"
        else:
            # Use filename without extension as caption
            self.captions = [f.stem.replace("_", " ") for f in self.image_files]
        
        self.transform = transform
        self.normalize_to_neg_one = normalize_to_neg_one

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        # Convert to tensor and normalize
        image = torch.tensor(np.array(image)).float() / 255.0
        
        if self.normalize_to_neg_one:
            image = 2 * image - 1
            
        image = rearrange(image, "h w c -> c h w")

        return image, caption


def setup_forget_style_data(data_method, forget_data_dir, remain_data_dir, batch_size,
                            image_size, interpolation='bicubic', normalize_to_neg_one=True):
    """
    Setup data loaders for style removal/forgetting tasks
    
    Args:
        forget_data_dir: Directory containing forget data
        remain_data_dir: Directory containing remain data
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        interpolation: Interpolation method for resizing
        normalize_to_neg_one: If True, normalize to [-1, 1] for diffusers
    
    Returns:
        Tuple of (forget_dataloader, remain_dataloader)
    """
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    # Setup forget dataset
    print(f"Loading datasets...")
    
    # Create lists of paths
    forget_data_path = []
    forget_prompt_path = []
    remain_data_path = []
    remain_prompt_path = []

    # CA uses captions.txt instead of prompts.txt
    text_name = 'prompts.txt' if data_method == 'erasediff' else 'captions.txt'
    for i in range(len(forget_data_dir)):
        forget_data_path.append(os.path.join(forget_data_dir[i], 'images.txt'))
        forget_prompt_path.append(os.path.join(forget_data_dir[i], text_name))
        remain_data_path.append(os.path.join(remain_data_dir[i], 'images.txt'))
        remain_prompt_path.append(os.path.join(remain_data_dir[i], text_name))
    
    # Setup forget dataset
    forget_set = StyleDataset(
        forget_data_path, 
        forget_prompt_path, 
        transform=transform,
        normalize_to_neg_one=normalize_to_neg_one
    )

    # Setup remain dataset
    remain_set = StyleDataset(
        remain_data_path, 
        remain_prompt_path, 
        transform=transform,
        normalize_to_neg_one=normalize_to_neg_one,
    )

    # Create Dataloaders
    forget_dl = DataLoader(forget_set, batch_size=batch_size, shuffle=True, drop_last=True)
    remain_dl = DataLoader(remain_set, batch_size=batch_size, shuffle=True, drop_last=True)

    return forget_dl, remain_dl


# CHANGE 7: Add setup function for class-based forgetting (from original erasediff)
def setup_forget_data(class_to_forget, batch_size, image_size, data_dir="data", 
                     interpolation='bicubic', normalize_to_neg_one=True):
    """
    Setup data loader for class-based forgetting
    
    Args:
        class_to_forget: Class index or name to forget
        batch_size: Batch size
        image_size: Image size
        data_dir: Base data directory
        interpolation: Interpolation method
        normalize_to_neg_one: If True, normalize to [-1, 1]
    
    Returns:
        Tuple of (dataloader, class_descriptions)
    """
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)
    
    # This is a placeholder - implement based on your specific dataset structure
    class_dir = os.path.join(data_dir, f"class_{class_to_forget}")
    
    if os.path.exists(os.path.join(class_dir, 'images.txt')):
        # Use text file format
        data_path = os.path.join(class_dir, 'images.txt')
        prompt_path = os.path.join(class_dir, 'prompts.txt')
        dataset = StyleDataset(
            data_path, 
            prompt_path, 
            transform=transform,
            normalize_to_neg_one=normalize_to_neg_one
        )
    else:
        # Use directory format
        dataset = ImageCaptionDataset(
            class_dir,
            transform=transform,
            normalize_to_neg_one=normalize_to_neg_one
        )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Example class descriptions (customize based on your classes)
    class_descriptions = [
        f"a photo of class {i}" for i in range(10)
    ]
    
    return dataloader, class_descriptions


def setup_remain_data(class_to_forget, batch_size, image_size, data_dir="data",
                     interpolation='bicubic', normalize_to_neg_one=True):
    """
    Setup data loader for remaining classes (all except forgotten class)
    
    Args:
        class_to_forget: Class index or name to exclude
        batch_size: Batch size
        image_size: Image size
        data_dir: Base data directory
        interpolation: Interpolation method
        normalize_to_neg_one: If True, normalize to [-1, 1]
    
    Returns:
        Tuple of (dataloader, class_descriptions)
    """
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)
    
    # Placeholder - implement based on your dataset
    remain_dir = os.path.join(data_dir, "remain_data")
    
    if os.path.exists(os.path.join(remain_dir, 'images.txt')):
        data_path = os.path.join(remain_dir, 'images.txt')
        prompt_path = os.path.join(remain_dir, 'prompts.txt')
        dataset = StyleDataset(
            data_path, 
            prompt_path, 
            transform=transform,
            normalize_to_neg_one=normalize_to_neg_one
        )
    else:
        dataset = ImageCaptionDataset(
            remain_dir,
            transform=transform,
            normalize_to_neg_one=normalize_to_neg_one
        )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Example class descriptions
    class_descriptions = [
        f"a photo of class {i}" for i in range(10)
    ]
    
    return dataloader, class_descriptions


# CHANGE 8: Add utility functions for diffusers
def collate_fn(batch):
    """
    Custom collate function for batching images and captions
    """
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return images, captions


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    Create a dataloader with optimal settings for diffusers training
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )