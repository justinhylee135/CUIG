import torch
import gc
from diffusers import StableDiffusionPipeline, DDPMScheduler


def get_param(net):
    """Get parameters for BOME algorithm"""
    new_param = []
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            new_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return new_param


def set_param(net, old_param):
    """Set parameters for BOME algorithm"""
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            param.copy_(old_param[j])
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return net


def setup_diffusers_model(base_model_dir, device):
    """Setup diffusers model components"""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_dir,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    unet = pipe.unet.to(device)
    scheduler = DDPMScheduler.from_pretrained(base_model_dir, subfolder="scheduler")
    
    vae.eval()
    text_encoder.eval()
    
    return unet, vae, text_encoder, tokenizer, scheduler


def encode_prompts(prompts, tokenizer, text_encoder, device):
    """Encode text prompts to embeddings"""
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    return text_embeddings


def encode_images(images, vae):
    """Encode images to latent space"""
    if images.min() >= 0 and images.max() <= 1:
        images = images * 2.0 - 1.0
    
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * 0.18215
    
    return latents


def get_trainable_params(unet, train_method):
    """Get trainable parameters based on training method"""
    train_param_names = []
    parameters = []
    
    for name, param in unet.named_parameters():
        if train_method == 'noxattn':
            if 'out' in name or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                train_param_names.append(name)
                parameters.append(param)
        elif train_method == 'selfattn':
            if 'attn1' in name:
                train_param_names.append(name)
                parameters.append(param)
        elif train_method == 'xattn':
            if 'attn2' in name:
                train_param_names.append(name)
                parameters.append(param)
        elif train_method == 'kv-xattn':
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                train_param_names.append(name)
                parameters.append(param)
        elif train_method == 'full':
            train_param_names.append(name)
            parameters.append(param)
        elif train_method == 'notime':
            if not ('out' in name or 'time_embed' in name):
                train_param_names.append(name)
                parameters.append(param)
        elif train_method == 'xlayer':
            if 'attn2' in name:
                if 'up_blocks.1' in name or 'up_blocks.2' in name:
                    train_param_names.append(name)
                    parameters.append(param)
        elif train_method == 'selflayer':
            if 'attn1' in name:
                if 'down_blocks.1' in name or 'down_blocks.2' in name:
                    train_param_names.append(name)
                    parameters.append(param)
    
    return train_param_names, parameters

def create_anchor_prompts(forget_prompts, concept, anchor, concept_type):
    """Create anchor prompts for BOME algorithm"""
    anchor_prompts = []
    for prompt in forget_prompts:
        for i in range(len(concept)):
            replace_with = "Photo" if anchor[i] == "Seed_Images" else anchor[i]
            a_prompt = prompt.replace(concept[i], replace_with)
            if a_prompt != prompt:
                break
            
        if a_prompt == prompt:
            print(f"Replacement unsuccessful for prompt '{prompt}' and anchor '{anchor}'")
            if concept_type == "object":
                a_prompt = f"A {anchor} image"
            else: 
                a_prompt = f"An image in {anchor} style"
        anchor_prompts.append(a_prompt)
    
    return anchor_prompts