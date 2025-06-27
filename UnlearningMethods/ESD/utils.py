# Standard Library
from PIL import Image
import ast 

# Third Party
import torch
import numpy as np
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt

def process_input_concepts(concept, concept_type):
    """Process input concepts based on the concept type"""
    # Store concept(s) as a list
    if "[" in concept:
        concepts = ast.literal_eval(concept)
    else:
        concepts = [concept]

    # Prompt templates based on concept type
    prompt_templates = {
        "style": lambda c: f'{c.replace("_", " ")} Style',
        "object": lambda c: f'An image of {c}',
        "celeb": lambda c: f'{c.replace("_", " ")}'
    }

    # Get the appropriate prompt function based on concept type
    prompt_fn = prompt_templates.get(concept_type)
    if not prompt_fn:
        raise ValueError(f"Unknown concept_type: {concept_type}")

    # Generate prompts for each concept
    return [prompt_fn(c) for c in concepts]
        
def load_pipeline(base_model_dir, device="cuda", dtype=torch.float32):
    """Load a diffusers pipeline from model path"""    
    # Load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Set scheduler to DDIM
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    
    return pipeline

def get_pipelines(model_path, unet_ckpt, devices):
    """Get original and trainable pipelines"""
    # Load frozen and esd pipeline
    print(f"Loading pipeline from {model_path}")
    frz_pipeline = load_pipeline(model_path, devices[1])
    pipeline = load_pipeline(model_path, devices[0])
    
    # Load UNet checkpoint if provided
    if unet_ckpt is not None:
        print(f"Loading UNet checkpoint from {unet_ckpt}...")
        unet_state_dict = torch.load(unet_ckpt, map_location=devices[0])
        frz_pipeline.unet.load_state_dict(unet_state_dict)
        pipeline.unet.load_state_dict(unet_state_dict)
        print(f"UNet checkpoint loaded successfully from {unet_ckpt}")
    
    frz_pipeline.unet.eval()
    frz_pipeline.unet.requires_grad_(False)
    
    return frz_pipeline, pipeline

def get_trainable_params(pipeline, train_method):
    updatable_params = []
    parameters = []
    for name, param in pipeline.unet.named_parameters():
        # Only cross-attention
        if train_method == 'xattn':
            if 'attn2' in name:
                updatable_params.append(name)
                parameters.append(param)
        # Everything but cross-attention
        elif train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                continue
            updatable_params.append(name)
            parameters.append(param)
        # Only self-attention
        elif train_method == 'selfattn':
            if 'attn1' in name:
                updatable_params.append(name)
                parameters.append(param)
        # All parameters
        elif train_method == 'full':
            updatable_params.append(name)
            parameters.append(param)
                
    return updatable_params, parameters

@torch.no_grad()
def sample_model(pipeline, prompt, negative_prompt="", height=512, width=512, 
                num_inference_steps=50, guidance_scale=7.5, eta=0.0, 
                latents=None, t_start=None, return_intermediates=False):
    """Sample the model using diffusers pipeline"""
    
    if t_start is not None:
        # For partial denoising (till time step t)
        # Convert t_start to the appropriate timestep for the scheduler
        pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = pipeline.scheduler.timesteps
        start_timestep = timesteps[t_start] if t_start < len(timesteps) else timesteps[0]
        
        # Encode text
        text_embeddings = pipeline._encode_prompt(
            prompt, device=pipeline.device, num_images_per_prompt=1, 
            do_classifier_free_guidance=guidance_scale > 1.0, negative_prompt=negative_prompt
        )
        
        # Prepare latents
        if latents is None:
            latents = torch.randn((1, 4, height // 8, width // 8), 
                                device=pipeline.device, dtype=pipeline.unet.dtype)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        # Partial denoising
        for i, t in enumerate(timesteps):
            if t < start_timestep:
                break
                
            # Expand latents if doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous timestep latent
            latents = pipeline.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
        
        return latents
    else:
        # Full generation
        with torch.autocast(device_type=pipeline.device.type):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                latents=latents,
                return_dict=True
            )
        return result.images[0] if not return_intermediates else result

def encode_text(text_encoder, tokenizer, prompt, device):
    """Encode text using the text encoder"""
    # Tokenize
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    # Encode text
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]
    
    return text_embeddings

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    # Load image
    image = Image.open(path).convert("RGB")
    
    # Resize and normalize
    tform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    
    return 2.*image - 1.

def moving_average(a, n=3):
    """Compute moving average of a 1D array"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path, word, n=100):
    """Plot the moving average of losses and save the figure"""
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)