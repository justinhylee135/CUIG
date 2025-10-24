# Standard Library Imports
import os
import sys
import copy
import math

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
    generate_unlearning_images,  
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
    upload_to_hub
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

# Forget Me Not Attention Controller and Processor Classes
class AttnController:
    def __init__(self) -> None:
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None

    def __call__(self, attn_prob, m_name):
        """
        attn_prob: (B*H, Q, K)
        self.concept_positions: (B, K) [bool]
        """
        if self.concept_positions is None:
            return

        B = self.concept_positions.shape[0]
        BH, Q, K = attn_prob.shape
        H = BH // B
        num_concept = int(self.concept_positions[0].sum().item())
        if num_concept == 0:
            # nothing to collect this step
            return

        # Build mask -> (B, 1, K) -> (B, Q, K) -> (B*H, Q, K)
        mask = self.concept_positions[:, None, :].expand(B, Q, K)         # (B,Q,K)
        mask = mask.repeat(H, 1, 1)                                       # (B*H,Q,K)
        mask = mask.to(device=attn_prob.device, dtype=torch.bool)

        # Select all attention to the concept tokens across all queries
        selected = attn_prob.masked_select(mask)                           # 1D
        # Each row corresponds to one (B,H,Q) slice over the concept keys
        # Length = (B*H*Q) * num_concept  -> reshape to (-1, num_concept)
        selected = selected.view(-1, num_concept)

        self.attn_probs.append(selected)
        self.logs.append(m_name)

    def set_concept_positions(self, concept_positions):
        # expect (B, K) bool; move to CPU/GPU as needed later
        self.concept_positions = concept_positions

    def loss(self):
        if len(self.attn_probs) == 0:
            # safe zero on the right device if possible
            dev = (self.concept_positions.device
                   if isinstance(self.concept_positions, torch.Tensor) else "cpu")
            return torch.tensor(0.0, device=dev)
        
        # Root Mean Squared Normalization
        concatenated = torch.cat(self.attn_probs, dim=0)
        loss = torch.sqrt((concatenated**2).mean())  
        return loss

    def zero_attn_probs(self):
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None

class MyCrossAttnProcessor:
    """Custom cross-attention processor that captures attention probabilities"""
    def __init__(self, attn_controller: "AttnController", module_name) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
    
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attn_controller(attention_probs, self.module_name)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def setup_attention_controller(unet):
    """Set up attention controller and processors for the UNet"""
    attn_controller = AttnController()
    module_count = 0
    for n, m in unet.named_modules():
        if n.endswith('attn2'):  # Only cross-attention layers
            m.set_processor(MyCrossAttnProcessor(attn_controller, n))
            module_count += 1
    logger.info(f"Set up attention controller for {module_count} cross-attention modules")
    return attn_controller

def get_concept_positions(batch, tokenizer, accelerator):
    """Extract concept positions from the batch for attention masking"""
    target_prompts = batch["target_prompts"]
    input_ids = batch["input_ids"]
    target_concepts = batch["target_concepts"]

    bsz, seq_len = input_ids.shape
    concept_positions = torch.zeros(bsz, seq_len, dtype=torch.bool, device=accelerator.device)
    
    for i, target_prompt in enumerate(target_prompts):
        # For FMN, the target prompt already contains the concept to forget
        # Tokenize the entire prompt to find concept positions
        prompt_ids = input_ids[i].cpu().tolist()
        target_concept = target_concepts[i]

        # Extract concept based on prompt structure
        if target_concept in target_prompt:
            concept = target_concept
        elif "style of" in target_prompt:
            concept = target_prompt.split("style of ")[-1].strip()
        elif "photo of" in target_prompt:
            concept = target_prompt.split("photo of ")[-1].strip()
        else:
            concept = target_prompt.strip()
        
        # Tokenize and find concept
        concept_ids = tokenizer(concept, add_special_tokens=False).input_ids
        
        match_found = False
        for j in range(len(prompt_ids) - len(concept_ids) + 1):
            if prompt_ids[j:j+len(concept_ids)] == concept_ids:
                concept_positions[i, j:j+len(concept_ids)] = True
                match_found = True
                break
            
        if not match_found:
            print(f"Warning: Tokenized Concept '{concept}' not found in prompt '{target_prompt}'")
        
        # Also mark the pooler token if needed
        if hasattr(args, 'use_pooler') and args.use_pooler:
            pooler_token_id = tokenizer("<|endoftext|>", add_special_tokens=False).input_ids[0]
            for j, tok_id in enumerate(prompt_ids):
                if tok_id == pooler_token_id:
                    concept_positions[i, j] = True
    
    return concept_positions
 
def main(args):
    # Overwrites existing ckpt only if flag is set '--overwrite_existing_ckpt'
    check_for_existing_ckpt(args)

    # Set up Logging and Accelerator
    accelerator, logger = setup_accelerator_and_logging(args)

    # Set seed 
    if args.seed is not None: set_seed(args.seed)
    
    # Setup concepts list
    setup_concepts_list(args)

    # Generate dataset for unlearning images if needed
    generate_unlearning_images(args, accelerator, logger)

    # Create output directory and huggingface repo (if selected)
    if accelerator.is_main_process:
        repo_id = setup_hub_repository(args)
    
    # Load and configure all models
    tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype = setup_models_and_tokenizer(args, accelerator)

    # Move VAE to device
    vae.to(accelerator.device, dtype=weight_dtype)

    # Configure training settings
    setup_training_configuration(args, unet, text_encoder, accelerator, logger, weight_dtype)

    # Set up attention controller for Forget Me Not method
    attn_controller = setup_attention_controller(unet)

    # Set up optimizer and parameters
    optimizer, modifier_token_id, params_to_optimize = setup_optimizer_and_params(args, unet, text_encoder, tokenizer)
    
    # Set up data and scheduler
    train_dataloader, lr_scheduler = setup_data_and_scheduler(args, tokenizer, accelerator, optimizer)

    setup_continual_enhancement(args, unet, text_encoder, tokenizer, accelerator.device)

    # Prepare training state
    prepared_components, global_step, first_epoch, progress_bar, resume_step = prepare_training_state(
        args, accelerator, unet, text_encoder, optimizer, train_dataloader, lr_scheduler
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
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=anchor_latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents (or use zeros if args.no_real_image is set)
                if hasattr(args, 'no_real_image') and args.no_real_image:
                    noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(anchor_latents, noise, timesteps)

                # Get the text embedding for conditioning
                token_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = text_encoder(token_ids)[0]
                
                # Get concept positions for attention masking
                concept_positions = get_concept_positions(batch, tokenizer, accelerator)
                concept_positions = concept_positions.to(device=accelerator.device, dtype=torch.bool)
                
                # Debug
                ids0 = batch["input_ids"][0]
                mask0 = concept_positions[0].to(device=ids0.device, dtype=torch.bool)
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None: mask0 = mask0 & (ids0 != eos_id)
                sel_ids = ids0[mask0].detach().cpu().tolist()
                sel_tokens = tokenizer.convert_ids_to_tokens(sel_ids)
                sel_text = tokenizer.convert_tokens_to_string(sel_tokens)

                # Set concept positions in the attention controller
                attn_controller.set_concept_positions(concept_positions)

                # Forward pass through UNet (this will capture attention probabilities)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute Forget Me Not attention loss
                loss = attn_controller.loss()
                
                # Store base loss for logging
                attention_loss = loss.detach().clone()

                # Add regularizers (keeping the original regularization structure)
                l1sp_loss = None
                l2sp_loss = None
                inverse_ewc_loss = None
                trajectory_loss = None
                if args.l1sp_weight > 0.0:
                    l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(unet, args.original_params)
                    loss = loss + l1sp_loss
                if args.l2sp_weight > 0.0:
                    l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(unet, args.original_params)
                    loss = loss + l2sp_loss
                if args.inverse_ewc_lambda > 0.0 and args.previous_aggregated_fisher is not None:
                    inverse_ewc_loss = args.inverse_ewc_lambda * calculate_inverse_ewc_loss(
                        unet, 
                        args.previous_aggregated_fisher, 
                        args.original_params, 
                        accelerator.device,
                        use_l2=args.inverse_ewc_use_l2
                    )
                    loss = loss + inverse_ewc_loss
                if args.trajectory_lambda > 0.0 and args.previous_aggregated_delta is not None:
                    trajectory_loss = args.trajectory_lambda * calculate_trajectory_loss(
                        unet,
                        args.previous_aggregated_delta,
                        args.original_params,
                        accelerator.device
                    )
                    loss = loss + trajectory_loss
                
                # Compute gradients
                accelerator.backward(loss)
                
                # Clear attention controller for next iteration
                attn_controller.zero_attn_probs()
                
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
            logs = {"total_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "t": timesteps[0].item()}
            logs["AttendToken[0]"] = f"'{sel_text}'"
            logs["targets[0]"] = f"'{batch['target_prompts'][0]}'"
            logs["attn_loss"] = attention_loss.item()
            if l1sp_loss is not None: logs["l1_loss"] = l1sp_loss.item()
            if l2sp_loss is not None: logs["l2_loss"] = l2sp_loss.item()
            if inverse_ewc_loss is not None: logs["ewc_loss"] = inverse_ewc_loss.item()
            if trajectory_loss is not None: logs["trajectory_loss"] = trajectory_loss.item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            iteration += 1

            # Simultaneous evaluation (keeping original evaluation logic)
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