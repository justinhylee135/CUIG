import os
import random
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils import encode_text, sample_model


class ReplayDataset:
    """Dataset for replay images and prompts"""
    def __init__(self, replay_dataset_dir, replay_prompts_path, image_size=512):
        self.image_size = image_size
        self.replay_images = []
        self.replay_prompts = []
        
        # Load replay prompts
        if replay_prompts_path and os.path.exists(replay_prompts_path):
            with open(replay_prompts_path, 'r') as f:
                self.replay_prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load replay images
        if replay_dataset_dir and os.path.exists(replay_dataset_dir):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            for file_path in Path(replay_dataset_dir).glob('**/*'):
                if file_path.suffix.lower() in image_extensions:
                    self.replay_images.append(str(file_path))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"Loaded {len(self.replay_images)} replay images and {len(self.replay_prompts)} replay prompts")
    
    def get_random_replay_data(self):
        """Get random replay image and prompt"""
        if not self.replay_images and not self.replay_prompts:
            return None, None
            
        # Get random image if available
        replay_image = None
        if self.replay_images:
            image_path = random.choice(self.replay_images)
            image = Image.open(image_path).convert('RGB')
            replay_image = self.transform(image)
        
        # Get random prompt if available
        replay_prompt = random.choice(self.replay_prompts) if self.replay_prompts else ""
        
        return replay_image, replay_prompt
    
    def has_data(self):
        """Check if replay dataset has any data"""
        return len(self.replay_images) > 0 or len(self.replay_prompts) > 0


def encode_images_to_latents(vae, images, device):
    """Encode images to latent space"""
    if images is None:
        return None
    
    # Move to device and ensure correct format
    if isinstance(images, torch.Tensor):
        images = images.to(device)
    else:
        images = torch.tensor(images).to(device)
    
    if len(images.shape) == 3:
        images = images.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    return latents


def compute_replay_loss(
    replay_dataset,
    esd_pipeline,
    frz_pipeline,
    timestep,
    timestep_idx,
    criteria,
    args
):
    """
    Compute replay retention loss.
    
    Args:
        replay_dataset: ReplayDataset instance
        esd_pipeline: ESD diffusion pipeline
        frz_pipeline: Frozen diffusion pipeline  
        timestep: Current timestep tensor
        timestep_idx: Current timestep index
        criteria: Loss criterion (MSE)
        args: Arguments containing device info and other params
        
    Returns:
        replay_loss: Computed replay loss tensor
        replay_prompt: The replay prompt used (for logging)
    """
    if not replay_dataset.has_data():
        return torch.tensor(0.0, device=args.devices[0]), ""
    
    # Get random replay data
    replay_image, replay_prompt = replay_dataset.get_random_replay_data()
    
    if not replay_prompt:
        return torch.tensor(0.0, device=args.devices[0]), ""
    
    if replay_image is not None:
        # Use provided replay image
        with torch.no_grad():
            # Encode image to latents
            replay_latents = encode_images_to_latents(esd_pipeline.vae, replay_image, args.devices[0])
            
            # Add noise to the replay latents
            noise = torch.randn_like(replay_latents)
            noisy_latents = esd_pipeline.scheduler.add_noise(replay_latents, noise, timestep.unsqueeze(0))
    else:
        # Generate latents from replay prompt
        latents = torch.randn((1, 4, args.image_size // 8, args.image_size // 8)).to(args.devices[0])
        latents = latents * esd_pipeline.scheduler.init_noise_sigma
        
        with torch.no_grad():
            # Generate noisy latents using replay prompt
            noisy_latents = sample_model(
                (frz_pipeline if args.sample_latent_from_frz_pipeline else esd_pipeline), 
                prompt=replay_prompt, 
                height=args.image_size, 
                width=args.image_size,
                num_inference_steps=args.ddim_steps, 
                guidance_scale=args.start_guidance, 
                latents=latents,
                t_until=timestep_idx
            )
    
    # Get frozen model prediction for replay prompt (target)
    with torch.no_grad():
        emb_replay = encode_text(frz_pipeline.text_encoder, frz_pipeline.tokenizer, replay_prompt, args.devices[1])
        noisy_latents_frz = noisy_latents.to(args.devices[1])
        timestep_tensor_frz = timestep.unsqueeze(0).to(args.devices[1])
        
        # Get target prediction from frozen model
        target_prediction = frz_pipeline.unet(noisy_latents_frz, timestep_tensor_frz, encoder_hidden_states=emb_replay).sample
    
    # Get ESD model prediction
    emb_replay_esd = encode_text(esd_pipeline.text_encoder, esd_pipeline.tokenizer, replay_prompt, args.devices[0])
    noisy_latents_esd = noisy_latents.to(args.devices[0])
    timestep_tensor_esd = timestep.unsqueeze(0).to(args.devices[0])
    
    esd_prediction = esd_pipeline.unet(noisy_latents_esd, timestep_tensor_esd, encoder_hidden_states=emb_replay_esd).sample
    
    # Retention loss: ESD model should predict same as frozen model for replay data
    target_prediction = target_prediction.to(args.devices[0])
    replay_loss = criteria(esd_prediction, target_prediction)
    
    return replay_loss, replay_prompt


def setup_replay_dataset(args):
    """
    Initialize replay dataset from arguments.
    
    Args:
        args: Arguments containing replay_dataset_dir and replay_prompts_path
        
    Returns:
        ReplayDataset instance
    """
    replay_dataset = ReplayDataset(args.replay_dataset_dir, args.replay_prompts_path, args.image_size)
    
    if replay_dataset.has_data():
        print(f"Replay enabled with loss weight {args.replay_loss_weight}")
    else:
        print("No replay data provided, running standard ESD training")
        
    return replay_dataset