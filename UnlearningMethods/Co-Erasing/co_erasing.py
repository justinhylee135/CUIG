import argparse
import math
import os
import random
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection
from utils.training_utils import get_training_params
from utils.model_utils import (
    load_unet, load_others,
    ImageProjModel, IPAdapter, get_attn_processor
)
from utils.diffusion_utils import (
    set_scheduler_device, denoise_to_text_timestep,
    predict_image_t_noise, predict_text_t_noise
)
from utils.helpers import to_same_device, save_model


transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {dtype_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-guided co-erasing training with IP-Adapter guidance."
    )
    parser.add_argument('--modality', type=str, choices=['text', 'image'], default='image')
    parser.add_argument('--train_method', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='/data/feiran/stable-diffusion-v1-5', help="Base SD checkpoint / repo.")
    parser.add_argument('--unet_ckpt_path', type=str, default=None)
    parser.add_argument('--image_encoder_path', type=str, default=None, help="Path or repo for the CLIP vision encoder.")
    parser.add_argument('--ip_adapter', type=str, default=None, help="Path to IP-Adapter checkpoint (.bin).")
    parser.add_argument('--save_path', type=str, default=None, help="Root directory for checkpoints.")
    parser.add_argument('--save_iter', type=int, default=500)
    parser.add_argument('--negative_guidance', type=float, default=1.0)
    parser.add_argument('--image', type=str, default=None, help="Folder or single image path for direct guidance.")
    parser.add_argument('--contrastive_image', type=str, default=None, help="Folder of positive/negative pairs.")
    parser.add_argument('--image_number', type=int, default=100)
    parser.add_argument('--noise_factor', type=float, default=0.0)
    parser.add_argument('--blur_factor', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--text_uncond", action='store_true')
    parser.add_argument("--text_guide", type=str, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--diff_method', type=str, default='diff', choices=['diff', 'l1', 'l2'])
    parser.add_argument('--use_average', action='store_true', help="Use averaged contrastive embedding.")
    return parser.parse_args()


def _resolve_image_encoder_path(args) -> str:
    if args.image_encoder_path:
        expanded = os.path.abspath(os.path.expanduser(args.image_encoder_path))
        if os.path.isdir(expanded):
            return expanded
        return args.image_encoder_path

    ckpt_dir = os.path.abspath(os.path.expanduser(args.ckpt_path))
    candidate = os.path.join(ckpt_dir, "image_encoder")
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        "Image encoder path not provided and default '<ckpt_path>/image_encoder' does not exist. "
        "Pass --image_encoder_path pointing to a valid folder or Hugging Face repo id."
    )


def _resolve_ip_adapter_path(args) -> str:
    search_candidates = []
    if args.ip_adapter:
        search_candidates.append(os.path.abspath(os.path.expanduser(args.ip_adapter)))

    ckpt_dir = os.path.abspath(os.path.expanduser(args.ckpt_path))
    search_candidates.extend(
        [
            os.path.join(ckpt_dir, "ip_adapter", "ip-adapter_sd15.bin"),
            os.path.join(ckpt_dir, "ip_adapter_sd15.bin"),
        ]
    )

    for candidate in search_candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "IP-Adapter checkpoint not found. Provide --ip_adapter pointing to 'ip-adapter_sd15.bin'."
    )


def train_image_mode(args):
    torch_dtype = get_torch_dtype(getattr(args, "torch_dtype", "bfloat16"))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    raw_devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    if not raw_devices:
        raw_devices = ['0']
    if len(raw_devices) == 1:
        raw_devices = raw_devices * 2

    def _normalize_device(token: str) -> torch.device:
        token = token.lower()
        if token == "cpu":
            return torch.device("cpu")
        if token.startswith("cuda"):
            return torch.device(token)
        return torch.device(f"cuda:{int(token)}")

    device_list = [_normalize_device(d) for d in raw_devices[:2]]
    device_1, device_2 = device_list
    print(f"[INFO] Using torch dtype '{torch_dtype}' with frozen UNet on {device_1} and trainable UNet on {device_2}")

    # Load models
    origin_unet = load_unet(
        args.ckpt_path,
        requires_grad=False,
        torch_dtype=torch_dtype,
        device=device_1,
    )
    unet = load_unet(
        args.ckpt_path,
        requires_grad=True,
        torch_dtype=torch_dtype,
        device=device_2,
    )

    if args.unet_ckpt_path:
        ckpt_path = os.path.abspath(os.path.expanduser(args.unet_ckpt_path))
        if not os.path.exists(ckpt_path):
            print(f"[WARN] UNet checkpoint not found at {ckpt_path}. Using base weights.")
        else:
            if ckpt_path.endswith(".safetensors"):
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = unet.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[INFO] Loaded UNet checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")
            origin_unet.load_state_dict(state_dict, strict=False)

    unet.train()
    origin_unet.eval()

    _, tokenizer, text_encoder, noise_scheduler, _ = load_others(
        args.ckpt_path,
        requires_grad=False,
        torch_dtype=torch_dtype,
        device=device_1,
    )
    text_encoder = text_encoder.to(device_1, dtype=torch_dtype)

    # Init optimizer
    _, parameters = get_training_params(unet, args.train_method)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_inference_steps = args.num_inference_steps

    noise_scheduler.set_timesteps(num_inference_steps)

    # Define save path
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    unet_save_path = args.save_path

    writer = SummaryWriter(save_dir)

    image_encoder_path = _resolve_image_encoder_path(args)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(origin_unet.device)
    image_encoder.requires_grad_(False)

    # Image projection models
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(device_2, dtype=torch_dtype)

    origin_image_proj_model = ImageProjModel(
        cross_attention_dim=origin_unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(device_1, dtype=torch_dtype)

    # Attention processors
    attn_procs = get_attn_processor(unet)
    for processor in attn_procs.values():
        processor.to(device_2, dtype=torch_dtype)
    unet.set_attn_processor(attn_procs)

    origin_attn_procs = get_attn_processor(origin_unet)
    for processor in origin_attn_procs.values():
        processor.to(device_1, dtype=torch_dtype)
    origin_unet.set_attn_processor(origin_attn_procs)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    origin_adapter_modules = torch.nn.ModuleList(origin_unet.attn_processors.values())
    ip_adapter_path = _resolve_ip_adapter_path(args)

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, ip_adapter_path)
    origin_ip_adapter = IPAdapter(origin_unet, origin_image_proj_model, origin_adapter_modules, ip_adapter_path)

    # Prepare text embeddings
    text_input_ids = tokenizer(args.prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    text_embeddings = text_encoder(text_input_ids.to(device_1))[0].to(device_2, dtype=torch_dtype)

    uncond_input_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
    uncond_text_embeddings = text_encoder(uncond_input_ids.to(device_1))[0].to(device_2, dtype=torch_dtype)

    # Collect image embeddings (direct or contrastive)
    use_direct_image = args.image is not None
    use_contrastive = args.contrastive_image is not None
    use_text_guide = args.text_guide is not None
    assert use_direct_image or use_contrastive, "Set at least one image guidance method!"

    if use_direct_image:
        image_list = _load_image_list(args.image, args.image_number)
        print(f"[INFO] Found {len(image_list)} images to erase from: {args.image}")

    if use_contrastive:
        concept_embeds = _collect_contrastive_embeddings(args, image_encoder)
        concept_embed_tensor = torch.mean(torch.stack(concept_embeds), dim=0)

    # Training loop
    set_scheduler_device(noise_scheduler, unet.device)
    for idx in tqdm(range(args.iterations), desc="[Image Training]"):
        optimizer.zero_grad()

        # Sample image embedding
        if use_direct_image:
            image_embeds = _sample_image_embedding(image_list, image_encoder)

        elif use_contrastive:
            image_embeds = _sample_contrastive_embedding(args, concept_embeds, concept_embed_tensor)

        image_embeds = image_embeds.to(device_2, dtype=torch_dtype)

        if use_text_guide:
            key = text_embeddings
            guided_image_embeds = image_embeds.to(device_1, dtype=torch_dtype)
            query = origin_ip_adapter.image_proj_model(guided_image_embeds)
            value = key
            
            attention_scores = torch.matmul(query.to(origin_unet.device), key.to(origin_unet.device).transpose(1, 2))
            # Scale the attention scores
            d_k = key.size(-1)  # embedding_dim
            scaled_attention_scores = attention_scores / math.sqrt(d_k)
            
            # Apply softmax to get the attention weights
            attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # Shape: [batch_size, 1, num_patches]
            
            # Compute the weighted sum of the image embeddings (weighted by attention)
            attended_image_embedding = torch.matmul(attention_weights, value.to(origin_unet.device))  # Shape: [batch_size, 1, embedding_dim]
            image_embeds = attended_image_embedding.to(device_2, dtype=torch_dtype)

        # Add noise to image embedding if specified
        if args.noise_factor > 0:
            noise = torch.rand_like(image_embeds)
            image_embeds = image_embeds + args.noise_factor * noise

        # Sample timestep
        timestep_idx = random.randrange(num_inference_steps)
        lower = int(timestep_idx * 1000 / num_inference_steps)
        upper = max(lower + 1, int((timestep_idx + 1) * 1000 / num_inference_steps))
        t_ddpm = torch.randint(lower, upper, (1,), device=device_2, dtype=torch.long)

        # Start from noise
        start_code = torch.randn((1, 4, 64, 64), device=device_2, dtype=torch_dtype)

        with torch.no_grad():
            z = denoise_to_text_timestep(unet, text_embeddings, timestep_idx, start_code, noise_scheduler)
            if args.blur_factor > 0:
                z = torchvision.transforms.functional.gaussian_blur(
                    z.to(dtype=torch.float32), kernel_size=args.blur_factor
                ).to(dtype=torch_dtype)

            cond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, text_embeddings, origin_ip_adapter, image_embeds)
            if args.text_uncond:
                uncond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, uncond_text_embeddings)
            else:
                uncond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, uncond_text_embeddings, origin_ip_adapter, image_embeds)

        cond_noise = predict_image_t_noise(z, t_ddpm, unet, text_embeddings, ip_adapter, image_embeds)

        # Compute loss
        cond_noise, uncond_origin_noise, cond_origin_noise = to_same_device(
            [cond_noise, uncond_origin_noise, cond_origin_noise],
            unet.device,
            dtype=torch_dtype,
        )

        loss = criterion(cond_noise, uncond_origin_noise - args.negative_guidance * (cond_origin_noise - uncond_origin_noise))

        loss.backward()
        optimizer.step()

        writer.add_scalar('image_training_loss', loss.item(), idx + 1)

        # if (idx + 1) % args.save_iter == 0:
        #     save_model(unet, unet_save_path, idx)
        #     print(f"[Checkpoint] Saved UNet at iteration {idx + 1}")

    # Always save the final model state, even if the loop never hit save_iter.
    save_model(unet, unet_save_path, idx=args.iterations - 1)
    print(f"[Checkpoint] Saved final UNet at iteration {args.iterations}")


# ========================
#  Support functions
# ========================

def _load_image_list(image_path, image_number):
    if os.path.isdir(image_path):
        all_images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('png', 'jpg'))]
        return random.sample(all_images, min(image_number, len(all_images)))
    else:
        return [image_path]


def _sample_image_embedding(image_list, image_encoder):
    image_path = random.choice(image_list)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
    return image_encoder(image_tensor).image_embeds


def _collect_contrastive_embeddings(args, image_encoder):
    concept_embeds = []
    pair_dirs = [os.path.join(args.contrastive_image, d) for d in os.listdir(args.contrastive_image)
                 if os.path.isdir(os.path.join(args.contrastive_image, d))]

    for pair_dir in tqdm(pair_dirs, desc="[Contrastive]"):
        pos = transform(Image.open(os.path.join(pair_dir, "positive.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)
        neg = transform(Image.open(os.path.join(pair_dir, "negative.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)

        pos_embed = image_encoder(pos).image_embeds
        neg_embed = image_encoder(neg).image_embeds

        if args.diff_method == "diff":
            embed = pos_embed - neg_embed
        elif args.diff_method == "l1":
            embed = torch.abs(pos_embed - neg_embed)
        elif args.diff_method == "l2":
            embed = (pos_embed - neg_embed) ** 2
        else:
            raise ValueError(f"Invalid diff_method: {args.diff_method}")

        concept_embeds.append(embed)

    return concept_embeds


def _sample_contrastive_embedding(args, concept_embeds, avg_embed):
    if args.use_average:
        return avg_embed
    else:
        return random.choice(concept_embeds)


if __name__ == "__main__":
    cli_args = parse_args()
    train_image_mode(cli_args)
