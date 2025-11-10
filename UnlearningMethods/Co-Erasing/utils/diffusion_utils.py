import torch

def denoise_to_text_timestep(unet, text_embeddings, t, start_code, noise_scheduler):
    scheduler_dtype = noise_scheduler.betas.dtype
    latents = start_code.to(device=unet.device, dtype=unet.dtype)
    text_embeddings = text_embeddings.to(device=unet.device, dtype=unet.dtype)
    timesteps = noise_scheduler.timesteps
    for i, step in enumerate(reversed(timesteps)):
        t_tensor = torch.tensor([step], dtype=torch.long, device=unet.device)
        with torch.no_grad():
            noise_pred = unet(latents, t_tensor, encoder_hidden_states=text_embeddings).sample
        latents_fp32 = latents.to(dtype=scheduler_dtype)
        noise_pred_fp32 = noise_pred.to(dtype=scheduler_dtype)
        latents = noise_scheduler.step(noise_pred_fp32, t_tensor, latents_fp32).prev_sample.to(dtype=unet.dtype)
        if i == t:
            break
    return latents

def predict_text_t_noise(z, t_ddpm, unet, text_embeddings):
    noise_pred = unet(
        z.to(device=unet.device, dtype=unet.dtype),
        t_ddpm.to(unet.device),
        encoder_hidden_states=text_embeddings.to(device=unet.device, dtype=unet.dtype),
        return_dict=False,
    )[0]
    return noise_pred

def predict_image_t_noise(z, t_ddpm, unet, text_embedding, ip_adapter, image_embeds, schedule=None):
    noise_pred = ip_adapter(
        z.to(device=unet.device, dtype=unet.dtype),
        t_ddpm.to(unet.device),
        text_embedding.to(device=unet.device, dtype=unet.dtype),
        image_embeds.to(device=unet.device, dtype=unet.dtype),
        schedule,
    )

    return noise_pred



def set_scheduler_device(scheduler, device):
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.one = scheduler.one.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
