# Third Party
import torch
from tqdm import tqdm
from diffusers import DDIMScheduler

# Local
from .helpers import process_anchor


def _create_scheduler(device, ddim_steps=50):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(ddim_steps, device=device)
    return scheduler, ddim_steps


def _init_grad_dict(unet):
    grad_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            grad_dict[name] = 0
    return grad_dict


def _init_averaged_grad_dict(unet):
    averaged_grad_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            averaged_grad_dict[name] = torch.zeros_like(param)
    return averaged_grad_dict


def _prepare_conditioning(
    unet,
    target,
    anchor,
    device,
    is_sdxl,
    text_encoder,
    tokenizer,
    pipe,
):
    latent_size = getattr(unet.config, "sample_size", 64)
    shape = [1, 4, latent_size, latent_size]
    target_prompt = [target]
    anchor_prompt = [anchor]

    if is_sdxl:
        c_target, uc, target_pooled, uc_pooled = pipe.encode_prompt(
            prompt=target_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        c_anchor, _, anchor_pooled, _ = pipe.encode_prompt(
            prompt=anchor_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        text_encoder_projection_dim = (
            pipe.text_encoder_2.config.projection_dim
            if pipe.text_encoder_2
            else int(target_pooled.shape[-1])
        )
        pixel_size = shape[-1] * 8  # convert latent spatial dim back to image size
        add_time_ids = pipe._get_add_time_ids(
            (pixel_size, pixel_size),
            (0, 0),
            (pixel_size, pixel_size),
            dtype=c_target.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)
        add_time_ids = add_time_ids.repeat(c_target.shape[0], 1)
        return {
            "c_target": c_target,
            "c_anchor": c_anchor,
            "uc": uc,
            "target_pooled": target_pooled,
            "anchor_pooled": anchor_pooled,
            "uc_pooled": uc_pooled,
            "add_time_ids": add_time_ids,
        }

    return {
        "c_target": encode_text_diffusers(text_encoder, tokenizer, target_prompt, device),
        "c_anchor": encode_text_diffusers(text_encoder, tokenizer, anchor_prompt, device),
        "uc": encode_text_diffusers(text_encoder, tokenizer, [""], device),
        "target_pooled": None,
        "anchor_pooled": None,
        "uc_pooled": None,
        "add_time_ids": None,
    }


def _cfg_added_cond_kwargs(conditioning, which, repeat=1):
    add_time_ids = conditioning["add_time_ids"]
    if add_time_ids is None:
        return None

    pooled_key = {
        "target": "target_pooled",
        "anchor": "anchor_pooled",
        "uc": "uc_pooled",
    }[which]
    text_embeds = conditioning[pooled_key]
    time_ids = add_time_ids if repeat == 1 else add_time_ids.repeat(repeat, 1)
    return {"text_embeds": text_embeds, "time_ids": time_ids}


def _sample_latents_at_timestep(
    t_idx,
    scheduler,
    timesteps,
    shape,
    unet,
    device,
    conditioning,
    is_sdxl,
    start_guidance,
):
    if t_idx == 0:
        latents = torch.randn(shape, device=device, dtype=unet.dtype)
        latents = latents * scheduler.init_noise_sigma
        return latents, timesteps[0]

    latents = torch.randn(shape, device=device, dtype=unet.dtype)
    latents = latents * scheduler.init_noise_sigma

    for step_idx in range(t_idx):
        timestep_tensor = timesteps[step_idx].unsqueeze(0).to(device)

        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps[step_idx])

        # Concatenate embeddings for CFG
        encoder_hidden_states = torch.cat([conditioning["uc"], conditioning["c_target"]])
        added_cond_kwargs = None
        if is_sdxl:
            added_cond_kwargs = {
                "text_embeds": torch.cat([conditioning["uc_pooled"], conditioning["target_pooled"]]),
                "time_ids": conditioning["add_time_ids"].repeat(2, 1),
            }

        # Predict noise
        noise_pred = unet(
            latent_model_input,
            timestep_tensor,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + start_guidance * (noise_pred_text - noise_pred_uncond)

        # Compute previous sample
        latents = scheduler.step(noise_pred, timesteps[step_idx], latents).prev_sample

    return latents, timesteps[t_idx]


def _compute_step_objective(
    unet,
    latents,
    timestep_tensor,
    conditioning,
    loss,
    is_sdxl,
    negative_guidance,
):
    with torch.no_grad():
        # Get anchor and uc noise prediction (without gradient)
        if is_sdxl:
            noise_unconditional = unet(
                latents,
                timestep_tensor,
                encoder_hidden_states=conditioning["uc"],
                added_cond_kwargs=_cfg_added_cond_kwargs(conditioning, "uc"),
            ).sample
            noise_anchor = unet(
                latents,
                timestep_tensor,
                encoder_hidden_states=conditioning["c_anchor"],
                added_cond_kwargs=_cfg_added_cond_kwargs(conditioning, "anchor"),
            ).sample
        else:
            noise_unconditional = unet(latents, timestep_tensor, encoder_hidden_states=conditioning["uc"]).sample
            noise_anchor = unet(latents, timestep_tensor, encoder_hidden_states=conditioning["c_anchor"]).sample

    with torch.enable_grad():
        # Predicted noise for the target prompt
        if is_sdxl:
            noise_target = unet(
                latents,
                timestep_tensor,
                encoder_hidden_states=conditioning["c_target"],
                added_cond_kwargs=_cfg_added_cond_kwargs(conditioning, "target"),
            ).sample
        else:
            noise_target = unet(latents, timestep_tensor, encoder_hidden_states=conditioning["c_target"]).sample

        # Compute loss (Feel free to add new ones here)
        if loss == "ConAbl":
            # Difference between predicted noise conditioned on target versus anchor
            return ((noise_target - noise_anchor.detach()) ** 2).sum()
        if loss == "ESD":
            # Difference between predicted noise conditioned on target versus negative CFG
            criteria = torch.nn.MSELoss()
            reverse_cfg_noise = noise_unconditional - (negative_guidance * (noise_anchor - noise_unconditional))
            return criteria(noise_target, reverse_cfg_noise.detach())
        
        # Unknonw loss type
        raise ValueError(f"Unsupported SelFT loss type: {loss}")


def _accumulate_param_grads(unet, grad_dict):
    for name, param in unet.named_parameters():
        if param.grad is not None:
            grad_dict[name] += param.grad.detach().clone()


def _print_grad_norms(grad_dict, target, anchor):
    print(f"'{len(grad_dict)}' Gradient L2 norms for '{target}' to '{anchor}':")
    for name, grad in grad_dict.items():
        l2_norm = torch.norm(grad).item()
        print(f"\t{name}: {l2_norm}")


def _average_grad_dicts(unet, all_grad_dicts, num_prompts):
    print(f"\nAveraging gradients across '{num_prompts}' prompts...")
    averaged_grad_dict = _init_averaged_grad_dict(unet)

    for grad_dict in all_grad_dicts:
        for name in averaged_grad_dict.keys():
            averaged_grad_dict[name] += grad_dict[name]

    for name in averaged_grad_dict.keys():
        averaged_grad_dict[name] = averaged_grad_dict[name] / num_prompts

    print(f"\n'{len(averaged_grad_dict)}' Final averaged L2 norms:")
    for name, grad in averaged_grad_dict.items():
        l2_norm = torch.norm(grad).item()
        print(f"\t{name}: {l2_norm:.6f}")

    return averaged_grad_dict


def selft_get_score(unet, target_concepts, anchor_concepts, loss, device, text_encoder, tokenizer, pipe=None):
    """Compute SelFT scores using diffusers components."""
    is_sdxl = getattr(unet.config, "addition_embed_type", None) == "text_time"
    if is_sdxl and pipe is None:
        raise ValueError(
            "SDXL UNet detected but no pipeline was provided. Pass the StableDiffusionXLPipeline instance via `pipe`."
        )

    scheduler, ddim_steps = _create_scheduler(device)

    # Create list of gradient dictionary
    all_grad_dicts = []

    # Inference parameters
    start_guidance = 3.0
    negative_guidance = 1.0

    # Anchor list can be a single anchor or a list of anchors
    anchor_concepts = process_anchor(anchor_concepts, target_concepts)

    # Iterate through each prompt and average gradients
    num_iterations = len(target_concepts)
    for prompt_idx in range(num_iterations):
        # Create grad dictionary of 0s but with same shape as UNet
        grad_dict = _init_grad_dict(unet)

        # Pull target-anchor pair
        target_text = target_concepts[prompt_idx]
        anchor_text = anchor_concepts[prompt_idx]

        with torch.no_grad():
            print(f"Iteration {prompt_idx + 1}/{num_iterations}")
            print(f"\nSelFT target: '{[target_text]}'")
            print(f"SelFT anchor: '{[anchor_text]}'")

            # Returns text embeddings for target, anchor, and empty string (CFG)
            conditioning = _prepare_conditioning(
                unet=unet,
                target=target_text,
                anchor=anchor_text,
                device=device,
                is_sdxl=is_sdxl,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                pipe=pipe,
            )

        # Prepare shape for sampling
        shape = [1, 4, 512 // 8, 512 // 8]

        # Get timesteps
        timesteps = scheduler.timesteps

        # Iterate through denoising timesteps
        progress_bar = tqdm(range(min(ddim_steps, len(timesteps))), desc="Computing Gradients")
        for t_idx in progress_bar:
            
            # Generate denoised latent up to timestep t_idx
            with torch.no_grad():
                latents, current_timestep = _sample_latents_at_timestep(
                    t_idx=t_idx,
                    scheduler=scheduler,
                    timesteps=timesteps,
                    shape=shape,
                    unet=unet,
                    device=device,
                    conditioning=conditioning,
                    is_sdxl=is_sdxl,
                    start_guidance=start_guidance,
                )

            # Convert timestep to tensor
            timestep_tensor = current_timestep.unsqueeze(0).to(device)

            # Enable gradient tracking
            latents = latents.detach().requires_grad_(True)

            # Prepare UNet for gradient tracking
            unet.zero_grad()

            assert latents.shape[0] == conditioning["c_target"].shape[0], "Batch size mismatch between latents and conditioning"
            
            # Compute loss
            obj = _compute_step_objective(
                unet=unet,
                latents=latents,
                timestep_tensor=timestep_tensor,
                conditioning=conditioning,
                loss=loss,
                is_sdxl=is_sdxl,
                negative_guidance=negative_guidance,
            )

            # Backpropagate
            obj.backward()
            progress_bar.set_postfix(loss=obj.item())

            # Accumulate gradients
            _accumulate_param_grads(unet, grad_dict)

            # Clean up gradients for next iteration
            unet.zero_grad()

        # Calculate and print the L2 norm for each item in grad_dict
        _print_grad_norms(grad_dict, target_text, anchor_text)

        # Append the gradient dictionary for this target-anchor pair
        all_grad_dicts.append(grad_dict)

    return _average_grad_dicts(unet, all_grad_dicts, num_iterations)

def encode_text_diffusers(text_encoder, tokenizer, prompt, device):
    """Encode text using diffusers text encoder."""
    if isinstance(prompt, str):
        prompt = [prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]

    return text_embeddings
