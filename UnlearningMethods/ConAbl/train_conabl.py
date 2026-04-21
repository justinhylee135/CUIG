# Standard Library Imports
import os
import sys
import copy

# Third Party Imports
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from diffusers.utils import check_min_version

# Local Imports
## Diffusion Pipeline
from src.model import (
    CustomDiffusionPipeline,
    setup_models_and_tokenizer
)

## Anchor Dataset
from src.data import (
    generate_anchor_images_if_needed,
    setup_data_and_scheduler
)

## Training
from src.utils import (
    check_for_existing_ckpt,
    setup_accelerator_and_logging,
    setup_concepts_list,
    setup_hf_repo,
    setup_training_configuration,
    setup_optimizer_and_params,
    prepare_training_state,
    setup_regularizers,
    print_training_settings_summary,
    isolate_target_concept_emb_gradients,
    get_params_to_clip,
    update_progress_and_checkpoint,
    upload_to_hf
)

## Parse Arguments
from src.args import parse_args

## Regularizers
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
### Simultaneous
from Regularizers.Simultaneous.simultaneous import sample_and_evaluate_ua, check_early_stopping
### Weight
from Regularizers.Weight.l1sp import calculate_l1sp_loss
from Regularizers.Weight.l2sp import calculate_l2sp_loss
### SelFT (Used in utils.py)
### Projection
from Regularizers.Projection.projection import (apply_gradient_projection)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")
 
def main(args):
    # Overwrites existing ckpt only if flag is set '--overwrite_existing_ckpt'
    check_for_existing_ckpt(args)

    # Set up Logging and Accelerator
    accelerator, logger = setup_accelerator_and_logging(args)

    # Set seed (From accelerator library)
    if args.seed is not None: set_seed(args.seed)
    
    # Setup concepts list
    setup_concepts_list(args)

    # Generate dataset for anchor images if needed
    generate_anchor_images_if_needed(args, accelerator, logger)

    # Create huggingface repo to save to if selected (Does not by default, must set --push_to_hub)
    if accelerator.is_main_process: repo_id = setup_hf_repo(args)
    
    # Load and configure all models
    tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype = setup_models_and_tokenizer(args, accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Configure training settings
    setup_training_configuration(args, unet, text_encoder, accelerator, logger, weight_dtype)

    # Set up optimizer and parameters
    optimizer, token_ids_to_unlearn = setup_optimizer_and_params(args, unet, text_encoder, tokenizer)
    
    # Set up data and scheduler
    anchor_dataloader, lr_scheduler, anchor_dataset = setup_data_and_scheduler(args, tokenizer, optimizer)

    # Set up Regularizers for Continual Unlearning
    setup_regularizers(args, unet, text_encoder, tokenizer, accelerator.device)

    # Move important diffusion components to accelerator
    optimizer, anchor_dataloader, lr_scheduler = accelerator.prepare(optimizer, anchor_dataloader, lr_scheduler)
    if args.parameter_group == "text-emb":
        text_encoder = accelerator.prepare(text_encoder)
    else:
        unet = accelerator.prepare(unet)

    # If we loaded a checkpoint, this gives us the number of completed optimization steps, epochs and iterations (after last epoch) by the ckpt
    num_completed_opt_steps, num_completed_epochs, num_completed_iterations_in_epoch, progress_bar = prepare_training_state(args, accelerator, anchor_dataloader)

    # Print training settings summary
    print_training_settings_summary(args, accelerator, anchor_dataloader, anchor_dataset)

    # Start Training: Start from the number of epochs we've already completed
    for epoch in range(num_completed_epochs, args.epochs):
        
        # Set to training mode
        if args.parameter_group == "text-emb":
            text_encoder.train()
        else:
            unet.train()

        for iteration_in_epoch, batch in enumerate(anchor_dataloader):
            
            # Skip completed iterations by ckpt
            if (
                args.resume_from_checkpoint # Check we've resumed from a ckpt
                and epoch == num_completed_epochs # Do this check for the first epoch
                and iteration_in_epoch < num_completed_iterations_in_epoch # If we're under the number of completed iterations by checkpoint
            ):
                # Update our tracker for number of optimization steps
                if iteration_in_epoch % args.gradient_accumulation_steps == 0: progress_bar.update(1)

                # Skip this iteration
                continue
            
            # Select model to accumulate gradients
            if args.parameter_group == "text-emb":
                accumulate_model = text_encoder
            else:
                accumulate_model = unet

            # Ensure gradients are synchronized across multiple GPUs
            with accelerator.accumulate(accumulate_model):

                # Convert images from anchor dataset to latent 
                # Note: batch["input_anchor_images"] will have two sets of anchor images if "--with_anchor_preservation"
                # The first set will be anchor images used for mapping target concept to the anchor concept
                # The second set will be anchor images used for standard diffusion training loss on the anchor concept (preserve anchor concept)
                anchor_latents = vae.encode(batch["input_anchor_images"].to(device=accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                anchor_latents = anchor_latents * vae.config.scaling_factor
                
                # Extract batch size
                bsz = anchor_latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(anchor_latents)
                
                # Sample a random timestep for each image from range [0, diffusion scheduler max timesteps]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=anchor_latents.device,
                )
                timesteps = timesteps.long()

                # Add timestep porportional amount of noise to the anchor latent
                noisy_anchor_latents = noise_scheduler.add_noise(anchor_latents, noise, timesteps)

                # Get the token ids for the target and anchor prompts
                # batch["input_target_prompt_ids"] may include both target and prompt token ids if "--with_anchor_preservation" is on (to reduce number of forward passes)
                token_target_and_anchor_prompt = batch["input_target_prompt_ids"].to(accelerator.device)
                token_anchor_prompt = batch["input_anchor_prompt_ids"].to(accelerator.device)

                # Get text embeddings for target and anchor prompts
                emb_target_and_anchor_prompt = text_encoder(token_target_and_anchor_prompt)[0]
                emb_anchor_prompt = text_encoder(token_anchor_prompt)[0]

                # Predict the artifically added noise conditioned on the target prompt (also anchor prompt too if "--with_anchor_preservation")
                pnoise_target_and_anchor = unet(noisy_anchor_latents, timesteps, emb_target_and_anchor_prompt).sample
                
                # Predict noise conditioned on anchor concept (used for anchor preservation loss)
                with torch.no_grad():
                    pnoise_anchor_sg = unet(
                        noisy_anchor_latents[: emb_anchor_prompt.size(0)],
                        timesteps[: emb_anchor_prompt.size(0)],
                        emb_anchor_prompt,
                    ).sample

                # If we're preserving the anchor concept, then we need to utilize the actual noise that was added to the anchor latent
                # If "--with_anchor_preservation" then anchor_latents actually had two sets from the dataloader (one for unlearning and one for anchor preservation)
                # We want the noise that was added to just the set of images for anchor preservation (second set)
                if args.with_anchor_preservation:
                    gtruth_noise_anchor = torch.chunk(noise, 2, dim=0)[1]
 
                # Compute Unlearning and Anchor Preservation Loss
                if args.with_anchor_preservation:
                    # Separate into target and anchor conditioned noise prediction (remember both were given anchor latent, only difference is text conditioning)
                    pnoise_target, pnoise_anchor = torch.chunk(pnoise_target_and_anchor, 2, dim=0)
                    
                    # Compute Unlearning loss: Map target to anchor
                    unlearning_loss = F.mse_loss(pnoise_target.float(), pnoise_anchor_sg.float(), reduction="none")
                    
                    # Extract the anchor image masks (in case anchor image was augmented)
                    anchor_image_mask = torch.chunk(batch["input_anchor_image_masks"], 2, dim=0)[0].to(accelerator.device)

                    # 1. Ignore loss for the pixels outside of images valid region (by multiplying by mask)
                    # 2. Sum over [channel, height, width] so now we only have dimension batch_size 
                    # 3. Get average loss per valid pixel (divide by anchor_image_mask.sum([1,2,3])
                    # 4. Average across the batch (.mean())
                    unlearning_loss = ((unlearning_loss * anchor_image_mask).sum([1, 2, 3]) / anchor_image_mask.sum([1, 2, 3])).mean()

                    # Compute Anchor Preservation loss: Map anchor to ground truth noise that was added to the anchor image
                    anchor_preservation_loss = F.mse_loss(pnoise_anchor.float(), gtruth_noise_anchor.float(), reduction="mean")

                    # Combined Unlearning and Anchor Preservation loss
                    loss = unlearning_loss + args.anchor_preservation_weight * anchor_preservation_loss
                else:
                    # No Anchor Preservation loss, so pnoise_target_and_anchor actually only had predicted noise of anchor latent conditioned on target prompt
                    pnoise_target = pnoise_target_and_anchor
                    
                    # Compute unlearning loss
                    unlearning_loss = F.mse_loss(pnoise_target.float(), pnoise_anchor_sg.float(), reduction="none")
                    
                    # 1. Ignore loss for the pixels outside of images valid region (by multiplying by mask)
                    # 2. Sum over [channel, height, width] so now we only have dimension batch_size 
                    # 3. Get average loss per valid pixel (divide by anchor_image_mask.sum([1,2,3])
                    # 4. Average across the batch (.mean())
                    anchor_image_mask = batch["input_anchor_image_masks"].to(accelerator.device)
                    unlearning_loss = ((unlearning_loss * anchor_image_mask).sum([1, 2, 3]) / anchor_image_mask.sum([1, 2, 3])).mean()

                    # The final loss is just the unlearning loss
                    loss = unlearning_loss

                # Separate out loss for ConAbl so we can log it
                # Because next we will combine our loss with our regularizer losses
                conabl_loss = loss.detach().clone()
                
                # Compute and apply regularizers
                ## Weight Regularizers
                l1sp_loss = None
                l2sp_loss = None

                # By the way, calculate_l(1|2)sp_loss will initalize args.original_params with params from this first iteration
                if args.l1sp_weight > 0.0:
                    l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(unet, args.original_params)
                    loss += l1sp_loss
                if args.l2sp_weight > 0.0:
                    l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(unet, args.original_params)
                    loss += l2sp_loss

                # Compute gradients
                accelerator.backward(loss)

                # Make gradient orthogonal to text embedding space of auxiliary concepts
                if args.with_gradient_projection:
                    apply_gradient_projection(
                        unet=unet,
                        aux_orth_proj=args.aux_orth_proj,
                        device=accelerator.device,
                        accelerator=accelerator
                    )
                    
                # Zero gradients for all token embeddings except the concept embeddings
                if args.parameter_group == "text-emb":
                    isolate_target_concept_emb_gradients(args, accelerator, text_encoder, tokenizer, token_ids_to_unlearn)
                    
                # Clip gradients if necessary
                if accelerator.sync_gradients:
                    params_to_clip = get_params_to_clip(args.parameter_group, text_encoder, unet)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                # Take an optimization step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar and save checkpoint if needed
            num_completed_opt_steps = update_progress_and_checkpoint(
                accelerator, progress_bar, num_completed_opt_steps, args,
                unet, text_encoder, tokenizer, token_ids_to_unlearn
            )

            # Update logs
            # Generally important things (loss, learning rate, denoising timestep)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "timestep": timesteps[0].item()}

            # For sanity check, log a target prompt from our batch
            logs["target_prompt[0]"] = f"'{batch['input_target_prompts'][0]}'"

            # Also log our anchor concept (was used for string replacement with target concept)
            if args.concept_type != "style": logs["anchor_concepts[0]"] = f"'{batch['anchor_concepts'][0]}'"

            # If we're usinng anchor preservation we have separate unlearning and anchor preservation loss
            if args.with_anchor_preservation:
                logs["u_loss"] = unlearning_loss.item()
                logs["a_loss"] = anchor_preservation_loss.item()

            # Log loss for regularizers
            if l1sp_loss is not None: logs["l1sp_loss"] = l1sp_loss.item()
            if l2sp_loss is not None: logs["l2sp_loss"] = l2sp_loss.item()

            # If we have regularizers, log the separate loss for just ConAbl
            if l1sp_loss or l2sp_loss: logs["conabl_loss"] = conabl_loss.item()

            # Update progress bar with logs
            progress_bar.set_postfix(**logs)

            # Update log for accelerator
            accelerator.log(logs, step=num_completed_opt_steps)

            # Regularizer: Simultaneous
            # Check whether to run evaluation check
            if args.eval_interval is not None and num_completed_opt_steps % args.eval_interval == 0:

                # Create copy of unet and text_encoder for evaluation (more memory but safer for downcasting to fp16)
                eval_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                eval_text_encoder = copy.deepcopy(accelerator.unwrap_model(text_encoder))

                # Downcast precision to fp16
                if weight_dtype == torch.float32: 
                    eval_unet.half()
                    eval_text_encoder.half()

                # Create diffusion pipeline for sampling using our copied UNet and text encoder
                sampling_diffusion_pipeline = CustomDiffusionPipeline.from_pretrained( 
                    args.base_model_dir, # Original diffusion model (before unlearning)
                    unet=eval_unet, # Current UNet undergoing unlearning 
                    text_encoder=eval_text_encoder, # Current text encoder undergoing unlearning
                    tokenizer=tokenizer, # Tokenizer (Never updated during unlearning)
                    revision=args.hf_revision, # Revision tag for hugging face 
                    modifier_token_id=token_ids_to_unlearn, # Tokens that we used for unlearning if we unlearned via text encoder 
                    torch_dtype=torch.float16, # Precision
                ).to(accelerator.device, torch_dtype=torch.float16)

                # Set UNet and Text Encoder to evaluation mode
                sampling_diffusion_pipeline.unet.eval()
                sampling_diffusion_pipeline.text_encoder.eval()
                
                # Sample and evaluate unlearned accuracy
                save_path = os.path.join(args.output_dir, "delta.bin")
                ua = sample_and_evaluate_ua(sampling_diffusion_pipeline, 
                                            args.concept_type, 
                                            num_completed_opt_steps, 
                                            save_path, 
                                            args.target_concepts, 
                                            device=accelerator.device, 
                                            eval_classifier_dir=args.eval_classifier_dir, 
                                            eval_prompt_dir=args.eval_prompt_dir,
                                            parameter_group=args.parameter_group)
                print(f"Optimization Step: '{num_completed_opt_steps}', Unlearned Accuracy: '{ua}'")
                
                # Tear down diffusion pipeline and clear unnecessary memory
                del sampling_diffusion_pipeline
                torch.cuda.empty_cache()
                
                # Update best Unlearning Accuracy so far, how many iterations since improvement, and whether to trigger early stopping
                args.best_ua, args.no_improvement_count, args.early_stop_triggered = check_early_stopping(
                    ua, # This iteration's unlearning accuracy
                    args.best_ua, # Best unlearning accuracy so far
                    args.no_improvement_count, # Counter for patience
                    args.eval_interval, # Evaluate every n steps
                    args.patience, # How long we can go without improving Unlearning Accuracy
                    args.stop_threshold # Unlearn Accuracy threshold to trigger early stopping (Default: 99)
                )
                
                # If we've reached the UA threshold or patience limit, break out of
                # the inner loop: `for iteration_in_epoch, batch in enumerate(anchor_dataloader)`.
                if args.early_stop_triggered:
                    print(f"Stopping early at optimization step '{num_completed_opt_steps}' with best UA: '{args.best_ua}'")
                    break

            # If we've reached max optimization steps, break out of the inner loop:
            # `for iteration_in_epoch, batch in enumerate(anchor_dataloader)`.
            if num_completed_opt_steps >= args.iterations:
                break

        # Break out of the outer epoch loop if either stopping condition is met:
        # `for epoch in range(num_completed_epochs, args.epochs)`.
        if (args.eval_interval and args.early_stop_triggered) or (num_completed_opt_steps >= args.iterations):
            break

    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()

    # Only for the main process
    if accelerator.is_main_process:

        # Ensure UNet is full precision
        unet = unet.to(torch.float32)

        # Create diffusion pipeline soley for the purpose of saving
        export_diffusion_pipeline = CustomDiffusionPipeline.from_pretrained(
            args.base_model_dir, 
            unet=accelerator.unwrap_model(unet), 
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.hf_revision,
            modifier_token_id=token_ids_to_unlearn,
        )

        # Save path for final ckpt
        final_ckpt_save_path = os.path.join(args.output_dir, "delta.bin")

        # Save to disk
        export_diffusion_pipeline.save_pretrained(final_ckpt_save_path, parameter_group=args.parameter_group, all=False)

        # Upload to hub if requested
        if args.push_to_hub and repo_id:
            upload_to_hf(args, repo_id)
    
    # End of unlearning
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
