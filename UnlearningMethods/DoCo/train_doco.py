# Standard Library Imports
import os
import sys
import copy

# Third Party Imports
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.utils import check_min_version
import itertools

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
    check_for_existing_ckpt,
    setup_accelerator_and_logging,
    setup_concepts_list,
    setup_hub_repository,
    setup_training_configuration,
    setup_optimizer_and_params,
    prepare_training_state,
    setup_continual_enhancement,
    print_training_settings_summary,
    zero_out_embedding_gradients_except_concept,
    get_params_to_clip,
    update_progress_and_checkpoint,
    run_validation_step,
    run_final_validation,
    upload_to_hub,
    adjust_gradient
)
from src.args import parse_args

## Continual Enhancements
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
### Simultaneous
from ContinualEnhancements.Simultaneous.sim_utils import sample_and_evaluate_ua, check_early_stopping
### Regularization
from ContinualEnhancements.Regularization.l1sp import calculate_l1sp_loss
from ContinualEnhancements.Regularization.l2sp import calculate_l2sp_loss
from torchvision.utils import save_image
import os
from ContinualEnhancements.Regularization.inverse_ewc import (  
    accumulate_fisher,
    calculate_inverse_ewc_loss,
    save_fisher_information
)
from ContinualEnhancements.Regularization.trajectory import (
    calculate_trajectory_loss,
    save_delta_to_path
)
### SelFT Imports Used in utils.py
### Projection
from ContinualEnhancements.Projection.gradient_projection import (
    apply_gradient_projection
)
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")

logger = get_logger(__name__)
 
def main(args):
    # Overwrites existing ckpt only if flag is set '--overwrite_existing_ckpt'
    check_for_existing_ckpt(args)

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
        repo_id = setup_hub_repository(args)
    
    # Load and configure all models
    tokenizer, text_encoder, vae, unet, discriminator, noise_scheduler, weight_dtype = setup_models_and_tokenizer(args, accelerator)
    
    # Set noise_scheduler timesteps
    noise_scheduler.set_timesteps(args.num_timesteps)
    print(f"Set noise scheduler timesteps to '{len(noise_scheduler.timesteps)}'")

    # Create frozen copy of UNet for DoCo
    frozen_unet = copy.deepcopy(unet)
    for param in frozen_unet.parameters():
        param.requires_grad = False
        
    # Move VAE to device
    vae.to(accelerator.device, dtype=weight_dtype)

    # Configure training settings
    setup_training_configuration(args, unet, text_encoder, accelerator, logger, weight_dtype)

    # Set up optimizer and parameters
    optimizer, optimizer_D, modifier_token_id, params_to_optimize = setup_optimizer_and_params(args, unet, discriminator, text_encoder, tokenizer)
    criterion_D = torch.nn.BCEWithLogitsLoss()
    
    # Set up data and scheduler
    train_dataloader, lr_scheduler, lr_scheduler_D = setup_data_and_scheduler(args, tokenizer, accelerator, optimizer, optimizer_D)

    setup_continual_enhancement(args, unet, text_encoder, tokenizer, accelerator.device)

    # Prepare training state
    prepared_components, global_step, first_epoch, progress_bar, resume_step = prepare_training_state(
        args, accelerator, unet, frozen_unet, discriminator, text_encoder, optimizer, optimizer_D, train_dataloader, lr_scheduler
    )

    # Print training settings summary
    print_training_settings_summary(args, accelerator, logger, train_dataloader)

    # Start Training
    iteration = 0
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
                if args.use_random_latent:
                    timesteps = torch.full(
                        (bsz,), 
                        999,
                        device=anchor_latents.device,
                        dtype=torch.long
                    )
                else:
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
                    pnoise_anchor_sg = frozen_unet(
                        noisy_anchor_latents[: emb_anchor.size(0)],
                        timesteps[: emb_anchor.size(0)],
                        emb_anchor,
                    ).sample
                    
                    # Get denoised anchor from frozen UNet - process each example individually
                    denoised_anchor_frozen_list = []
                    for i in range(pnoise_anchor_sg.shape[0]):
                        denoised_output = noise_scheduler.step(
                            pnoise_anchor_sg[i:i+1],
                            timesteps[i],
                            noisy_anchor_latents[i:i+1]
                        )
                        denoised_anchor_frozen_list.append(denoised_output.pred_original_sample)
                        
                        # DEBUG: Save anchor images
                        # anchor_image = vae.decode(denoised_output.pred_original_sample / vae.config.scaling_factor).sample
                        # output_dir = os.path.join(args.output_dir, "anchor_images")
                        # os.makedirs(output_dir, exist_ok=True)
                        # save_path = os.path.join(output_dir, f"anchor_{i}.png")
                        # save_image(anchor_image[0].detach().cpu().clamp(0, 1), save_path)
                        
                    denoised_anchor_frozen = torch.cat(denoised_anchor_frozen_list, dim=0)


                # Get the target for the selected unlearning loss
                if args.loss_type_reverse == "model-based":
                    if args.with_prior_preservation:
                        gtruth_noise_anchor = torch.chunk(noise, 2, dim=0)[1]
                else:
                    # Other implementations removed
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                    
                # Split predictions for target and anchor
                if args.with_prior_preservation:
                    pnoise_target, pnoise_anchor = torch.chunk(pnoise_target_and_anchor, 2, dim=0)
                    gtruth_noise_anchor = torch.chunk(noise, 2, dim=0)[1]
                    timesteps_target, timesteps_anchor = torch.chunk(timesteps, 2, dim=0)
                    noisy_latents_target, noisy_latents_anchor = torch.chunk(noisy_anchor_latents, 2, dim=0)
                else:
                    pnoise_target = pnoise_target_and_anchor
                    timesteps_target = timesteps
                    noisy_latents_target = noisy_anchor_latents
               
                # Get denoised target from trainable UNet - process each example individually
                denoised_target_list = []
                for i in range(pnoise_target.shape[0]):
                    denoised_output = noise_scheduler.step(
                        pnoise_target[i:i+1],
                        timesteps_target[i],
                        noisy_latents_target[i:i+1]
                    )
                    denoised_target_list.append(denoised_output.pred_original_sample)
                    
                    # DEBUG: Save target images
                    # target_image = vae.decode(denoised_output.pred_original_sample / vae.config.scaling_factor).sample
                    # output_dir = os.path.join(args.output_dir, "target_images")
                    # os.makedirs(output_dir, exist_ok=True)
                    # save_path = os.path.join(output_dir, f"target_{i}.png")
                    # save_image(target_image[0].detach().cpu().clamp(0, 1), save_path)
                                    
                denoised_target = torch.cat(denoised_target_list, dim=0)
                
                # Train Discriminator
                if args.discrim_pixel_space:
                    # Decode latents to pixel space for discrimination
                    with torch.no_grad():
                        # Decode anchor (frozen UNet output)
                        anchor_pixels = vae.decode(denoised_anchor_frozen / vae.config.scaling_factor).sample
                        anchor_pixels = anchor_pixels.detach()
                        
                        # Decode target (trainable UNet output) for discriminator training
                        target_pixels_detached = vae.decode(denoised_target.detach() / vae.config.scaling_factor).sample
                        target_pixels_detached = target_pixels_detached.detach()      

                    # Classify anchor (from frozen UNet) as 0
                    anchor_class = discriminator(anchor_pixels).squeeze(1)
                    loss_anchor_D = criterion_D(anchor_class, torch.zeros_like(anchor_class))
                    
                    # Classify target (from trainable UNet) as 1
                    target_class = discriminator(target_pixels_detached).squeeze(1)
                    loss_target_D = criterion_D(target_class, torch.ones_like(target_class))                  
                else:    
                    # Classify anchor (from frozen UNet) as 0
                    anchor_class = discriminator(denoised_anchor_frozen.detach()).squeeze(1)
                    loss_anchor_D = criterion_D(anchor_class, torch.zeros_like(anchor_class))
                    
                    # Classify target (from trainable UNet) as 1
                    target_class = discriminator(denoised_target.detach()).squeeze(1)
                    loss_target_D = criterion_D(target_class, torch.ones_like(target_class))
                
                # Combined discriminator loss
                loss_D = loss_anchor_D + loss_target_D
                
                # Count number of correct predictions
                with torch.no_grad():
                    # Using logit sign avoids an extra sigmoid
                    anchor_correct = (anchor_class < 0).sum()
                    target_correct = (target_class > 0).sum()
                    D_correct = (anchor_correct + target_correct).item()
                    D_total = anchor_class.numel() + target_class.numel()
                    D_acc = D_correct / D_total
    
                # Update discriminator
                accelerator.backward(loss_D)
                optimizer_D.step()
                lr_scheduler_D.step()
                optimizer_D.zero_grad()
                      
                # Train UNet
                loss_UNet = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                
                # Retention loss (if using prior preservation)
                retention_loss = None
                if args.with_prior_preservation:
                    retention_loss = F.mse_loss(
                        pnoise_anchor.float(), 
                        gtruth_noise_anchor.float(), 
                        reduction="mean"
                    )
                    loss_UNet = loss_UNet + args.prior_loss_weight * retention_loss
                
                # Function used by adjust_gradient (ignore)
                def norm_grad():
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(text_encoder.parameters())
                            if args.parameter_group == "embedding"
                            else itertools.chain(
                                [x[1] for x in unet.named_parameters() if ("attn2" in x[0])]
                            )
                            if args.parameter_group == "cross-attn"
                            else itertools.chain(unet.parameters())
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Generator loss (after warmup)
                generator_loss = None
                if global_step > args.discrim_num_warmup_steps:
                    if args.discrim_pixel_space:
                        # Decode target for generator loss (need fresh decode for gradient flow)
                        with torch.no_grad():
                            target_pixels_gen = vae.decode(denoised_target / vae.config.scaling_factor).sample
                            target_pixels_gen = target_pixels_gen.detach()
                        
                        # Train UNet to fool discriminator (make target look like anchor)
                        target_class_gen = discriminator(target_pixels_gen).squeeze(1)
                    else:
                        # Original latent space
                        target_class_gen = discriminator(denoised_target).squeeze(1)
                    generator_loss = criterion_D(target_class_gen, torch.zeros_like(target_class_gen))
                    loss_UNet = loss_UNet + generator_loss
                    
                    if args.gradient_clip:
                        adjust_gradient(unet, optimizer, accelerator, norm_grad, generator_loss, retention_loss, lambda_=1)
                    else:
                        accelerator.backward(loss_UNet)
                   
                # Regularizers
                doco_loss = loss_UNet.detach().clone()
                l1sp_loss = None
                l2sp_loss = None
                inverse_ewc_loss = None
                trajectory_loss = None
                
                if args.l1sp_weight > 0.0:
                    l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(unet, args.original_params)
                    accelerator.backward(l1sp_loss, retain_graph=True)
                    
                if args.l2sp_weight > 0.0:
                    l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(unet, args.original_params)
                    accelerator.backward(l2sp_loss, retain_graph=True)

                if args.inverse_ewc_lambda > 0.0 and args.previous_aggregated_fisher is not None:
                    inverse_ewc_loss = args.inverse_ewc_lambda * calculate_inverse_ewc_loss(
                        unet, 
                        args.previous_aggregated_fisher, 
                        args.original_params, 
                        accelerator.device,
                        use_l2=args.inverse_ewc_use_l2
                    )
                    accelerator.backward(inverse_ewc_loss, retain_graph=True)
                    
                if args.trajectory_lambda > 0.0 and args.previous_aggregated_delta is not None:
                    trajectory_loss = args.trajectory_lambda * calculate_trajectory_loss(
                        unet,
                        args.previous_aggregated_delta,
                        args.original_params,
                        accelerator.device
                    )
                    accelerator.backward(trajectory_loss, retain_graph=True)
                
                # Accumulate Fisher for this unlearning run (only used next unlearning run)
                if args.save_fisher_path is not None:
                    args.current_fisher = accumulate_fisher(unet, args.current_fisher)
                
                # Make gradient orthogonal to text embedding space of anchor concepts
                if args.with_gradient_projection:
                    apply_gradient_projection(
                        model=unet,
                        filtered_embedding_matrix=args.anchor_embeddings_matrix,
                        device=accelerator.device,
                        accelerator=accelerator
                    )
                    
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
            logs = {"total_loss": loss_UNet.detach().item(), "t": timesteps[0].item()}
            logs["loss_D"] = loss_D.item()
            logs["D_acc"] = round(D_acc * 100, 2)
            logs["targets[0]"] = f"'{batch['target_prompts'][0]}'"
            if args.concept_type=="object": logs["anchors[0]"] = f"'{batch['anchors'][0]}'"
            if args.with_prior_preservation and (global_step > args.discrim_num_warmup_steps+1):
                logs["u_loss"] = generator_loss.item()
                logs["r_loss"] = retention_loss.item()
            if l1sp_loss is not None: logs["l1_loss"] = l1sp_loss.item()
            if l2sp_loss is not None: logs["l2_loss"] = l2sp_loss.item()
            if inverse_ewc_loss is not None: logs["ewc_loss"] = inverse_ewc_loss.item()
            if trajectory_loss is not None: logs["trajectory_loss"] = trajectory_loss.item()
            if len(logs) > (4 + int(args.concept_type=="object")): logs["doco_loss"] = doco_loss.item()
            progress_bar.set_postfix(**logs)
            if global_step < args.discrim_num_warmup_steps:
                progress_bar.set_description(f"Discrim. Steps ('{args.discrim_num_warmup_steps}' Warmup)")
            else:
                progress_bar.set_description("UNet Steps")
                
            accelerator.log(logs, step=global_step)
            iteration += 1

            # Simultaneous
            if args.eval_every is not None and iteration % args.eval_every == 0:
                # Create sampling pipeline (Use fp16 for quicker sampling)
                eval_unet = copy.deepcopy(accelerator.unwrap_model(unet))
                eval_text_encoder = copy.deepcopy(accelerator.unwrap_model(text_encoder))
                if weight_dtype == torch.float32: 
                    eval_unet.half()
                    eval_text_encoder.half()
                pipeline = CustomDiffusionPipeline.from_pretrained(
                    args.base_model_dir,
                    unet=eval_unet,
                    text_encoder=eval_text_encoder,
                    tokenizer=tokenizer,
                    revision=args.revision,
                    modifier_token_id=modifier_token_id,
                    torch_dtype=torch.float16,
                ).to(accelerator.device, torch_dtype=torch.float16)
                pipeline.unet.eval()
                pipeline.text_encoder.eval()
                
                # Sample and evaluate unlearned accuracy
                save_path = os.path.join(args.output_dir, "delta.bin")
                ua = sample_and_evaluate_ua(pipeline, args.concept_type, iteration, save_path, args.prompt_list, 
                                            None, accelerator.device, args.classifier_dir)
                print(f"Iteration '{iteration}', Unlearned Accuracy: '{ua}'")
                
                # Tear down pipeline
                del pipeline
                torch.cuda.empty_cache()
                
                # Check for early stopping
                args.best_ua, args.no_improvement_count, args.stop_training = check_early_stopping(
                    ua, args.best_ua, args.no_improvement_count, args.eval_every, args.patience
                )
                
                if args.stop_training:
                    logger.info(f"Stopping training early at iteration {iteration} with best UA: {args.best_ua}")
                    # Breaks out of inner for loop (batch)
                    break

            # Check if we reached the maximum number of training steps
            if global_step >= args.max_train_steps:
                break

        if args.eval_every and args.stop_training:
            # Breaks out of outer for loop (epoch)
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
        
        # Save Fisher Information
        if args.save_fisher_path is not None and args.current_fisher:
            save_fisher_information(args.current_fisher, args.save_fisher_path, iteration, args.previous_aggregated_fisher)
        
        # Save delta
        if args.save_delta_path is not None:
            save_delta_to_path(unet, args.original_params, args.save_delta_path, args.previous_aggregated_delta)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)