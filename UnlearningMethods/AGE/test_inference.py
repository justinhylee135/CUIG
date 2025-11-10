import torch
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# Load each component manually
pipe = StableDiffusionPipeline.from_pretrained(
    "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# Replace the UNet model manually
unet = UNet2DConditionModel.from_pretrained(
    "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas",
    subfolder="unet",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")
pipe.unet = unet

og_sd = pipe.unet.state_dict().copy()
sd_ckpt = torch.load("/fs/scratch/PAS2099/lee.10369/CUIG/age/models/independent/base/class/debug/Cats.pt", map_location="cuda")

total_l2 = 0
for key in sd_ckpt:
    l2 = torch.norm(og_sd[key] - sd_ckpt[key])
    total_l2 += l2.item() ** 2
    # print(f"Key: {key}, L2 Difference: {l2.item()}")
print(f"Total L2 Difference: {total_l2 ** 0.5}")

loadCkpt = False
if loadCkpt:
    missing, expected = pipe.unet.load_state_dict(sd_ckpt, strict=False)
    print(f"Loaded keys: {len(sd_ckpt)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(expected)}")

prompt = "Rust Style"
save_path = "cat_image.png"
print(f"Generating image for prompt: '{prompt}' to '{save_path}'")
pipe(prompt=prompt, num_inference_steps=100, guidance_scale=7.5).images[0].save(save_path)