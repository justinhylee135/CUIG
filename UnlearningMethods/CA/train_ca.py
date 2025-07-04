# Standard Library Imports
import os

# Third Party Imports
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.utils import check_min_version

# Local Imports
from src.model import (
    CustomDiffusionPipeline,
    setup_models_and_tokenizer
)
from src.data import (
    generate_anchor_images_if_needed,
    setup_data_and_scheduler
)
from src.utils import (
    setup_accelerator_and_logging,
    setup_concepts_list,
    setup_hub_repository,
    setup_training_configuration,
    setup_optimizer_and_params,
    prepare_training_state,
    zero_out_embedding_gradients_except_concept,
    get_params_to_clip,
    update_progress_and_checkpoint,
    run_validation_step,
    run_final_validation,
    upload_to_hub
)
from src.args import parse_args

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")

logger = get_logger(__name__)
 
def main(args):
    # Set up Logging and Accelerator
    accelerator, logger = setup_accelerator_and_logging(args)

    # Set seed 
    if args.seed is not None: set_seed(args.seed)
    
    # Setup concepts list
    setup_concepts_list(args)

    # Generate dataset for anchor images if needed
    generate_anchor_images_if_needed(args, accelerator, logger)

    # Create output directory and huggingface repo (if selected)
    if accelerator.is_main_process:
        if args.output_dir is not None: os.makedirs(args.output_dir, exist_ok=True)
        repo_id = setup_hub_repository(args)
    
    # Load and configure all models
    tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype = setup_models_and_tokenizer(args, accelerator)

    # Move VAE to device
    vae.to(accelerator.device, dtype=weight_dtype)

    # Configure training settings
    setup_training_configuration(args, unet, text_encoder, accelerator, logger, weight_dtype)

    # Set up optimizer and parameters
    optimizer, modifier_token_id, params_to_optimize = setup_optimizer_and_params(args, unet, text_encoder, tokenizer)
    
    # Set up data and scheduler
    train_dataloader, lr_scheduler = setup_data_and_scheduler(args, tokenizer, accelerator, optimizer)

    # Prepare training state
    prepared_components, global_step, first_epoch, progress_bar, resume_step = prepare_training_state(
        args, accelerator, logger, unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Retention Loss
    if args.with_prior_preservation: 
        print(f"Using retention loss with weight '{args.prior_loss_weight}'")
    else:
        print("Not using retention loss, only unlearning loss will be computed.")

    # Target and Anchor Concept
    target, anchor = args.caption_target.split("+")
    if anchor == "*":
        print(f"Mapping target concept '{target}' to full anchor prompts in '{args.class_prompt}'.")
    else:
        print(f"Mapping target concept '{target}' to anchor '{anchor}' via replacement using prompts '{args.class_prompt}'.")

    # Start Training
    for epoch in range(first_epoch, args.num_train_epochs):
        (text_encoder if args.parameter_group == "embedding" else unet).train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming from a checkpoint
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # Accumulate gradients
            accumulate_model = unet if args.parameter_group != "embedding" else text_encoder
            with accelerator.accumulate(accumulate_model):
                # Convert images from anchor dataset to latent space
                anchor_latents = vae.encode(
                    batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                ).latent_dist.sample()
                anchor_latents = anchor_latents * vae.config.scaling_factor
                bsz = anchor_latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(anchor_latents)
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=anchor_latents.device,
                )
                timesteps = timesteps.long()

                # Add t timestep noise to the anchor latent
                noisy_anchor_latents = noise_scheduler.add_noise(anchor_latents, noise, timesteps)

                # Get the text embedding for the target and anchor concept
                token_target_and_anchor = batch["input_ids"].to(accelerator.device)
                token_anchor = batch["input_anchor_ids"].to(accelerator.device)
                emb_target_and_anchor = text_encoder(token_target_and_anchor)[0]
                emb_anchor = text_encoder(token_anchor)[0]


                # Predict the noise residual
                pnoise_target_and_anchor = unet(noisy_anchor_latents, timesteps, emb_target_and_anchor).sample
                
                # Predict noise conditioned on anchor concept (used for retention loss)
                with torch.no_grad():
                    pnoise_anchor_sg = unet(
                        noisy_anchor_latents[: emb_anchor.size(0)],
                        timesteps[: emb_anchor.size(0)],
                        emb_anchor,
                    ).sample

                # Get the target for the selected unlearning loss
                if args.loss_type_reverse == "model-based":
                    if args.with_prior_preservation:
                        gtruth_noise_anchor = torch.chunk(noise, 2, dim=0)[1]
                else:
                    # Other implementations removed
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
 

                # Compute Unlearning and Retention Loss
                if args.with_prior_preservation:
                    # Separate into target and anchor conditioned noise
                    pnoise_target, pnoise_anchor = torch.chunk(pnoise_target_and_anchor, 2, dim=0)
                    
                    # Compute Unlearning loss: Map target to anchor
                    unlearning_loss = F.mse_loss(
                        pnoise_target.float(), pnoise_anchor_sg.float(), reduction="none"
                    )
                    
                    # Define mask for only target portion
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0].to(accelerator.device)
                    unlearning_loss = ((unlearning_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    # Compute Retention loss: Map anchor to ground truth noise
                    retention_loss = F.mse_loss(
                        pnoise_anchor.float(), gtruth_noise_anchor.float(), reduction="mean"
                    )

                    # Combined unlearning and retention loss
                    loss = unlearning_loss + args.prior_loss_weight * retention_loss
                else:
                    # No retention loss, so only the target noise was predicted
                    pnoise_target = pnoise_target_and_anchor
                    
                    # Compute unlearning loss
                    loss = F.mse_loss(
                        pnoise_target.float(), pnoise_anchor_sg.float(), reduction="none"
                    )
                    
                    # Apply mask
                    mask = batch["mask"].to(accelerator.device)
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                accelerator.backward(loss)
                
                # Zero gradients for all token embeddings except the concept embeddings
                if args.parameter_group == "embedding":
                    zero_out_embedding_gradients_except_concept(
                        args, accelerator, text_encoder, tokenizer, modifier_token_id
                    )
                    
                # Clip gradients if necessary
                if accelerator.sync_gradients:
                    params_to_clip = get_params_to_clip(args, text_encoder, unet)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                # Take an optimization step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar and save checkpoint if needed
            global_step = update_progress_and_checkpoint(
                accelerator, progress_bar, global_step, args, logger,
                unet, text_encoder, tokenizer, modifier_token_id
            )

            # Update logs
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "timestep": timesteps[0].item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Check if we reached the maximum number of training steps
            if global_step >= args.max_train_steps:
                break
        
        # Run validation step if enabled by args.turn_on_validation
        run_validation_step(
            accelerator, args, logger, global_step, epoch,
            unet, text_encoder, tokenizer, modifier_token_id
        )

    # Finalize training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the final model
        unet = unet.to(torch.float32)
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.base_model_dir,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
            modifier_token_id=modifier_token_id,
        )
        save_path = os.path.join(args.output_dir, "delta.bin")
        pipeline.save_pretrained(save_path, parameter_group=args.parameter_group)

        # Run final validation if enabled
        images = None
        if args.turn_on_validation and args.validation_prompt and args.num_validation_images > 0:
            images = run_final_validation(
                args, accelerator, pipeline, epoch
            )

        # Upload to hub if requested
        if args.push_to_hub and repo_id:
            upload_to_hub(args, repo_id, images)
            
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)