# Standard Library Imports
import os

# Third Party Imports
import torch
from tqdm.auto import tqdm

# Local Imports
from src.args import parse_args
from src.model import (
    get_esd_trainable_parameters,
    load_sdxl_models,
    save_esd_checkpoint,
)
from src.training import prepare_conditioning_cache, run_esd_training_step
from src.utils import (
    apply_gradient_projection,
    check_for_existing_ckpt,
    cleanup_hooks,
    configure_learning_rate,
    prepare_concept_prompts,
    print_training_summary,
    set_seed,
    setup_regularizers,
)


def main(args):
    """Train an ESD-SDXL checkpoint with CUIG-style setup and regularizers."""
    # Check output directory and path
    check_for_existing_ckpt(args)

    # Load in concepts for unlearning process
    prepare_concept_prompts(args)

    # Set learning rate
    configure_learning_rate(args)
    
    # Set Seed
    set_seed(args.seed)


    # Load two copies of the denoising UNet, one to train and one to keep frozen
    pipe, base_unet, esd_unet, torch_dtype = load_sdxl_models(args)

    # Return parameters to train
    trainable_names, trainable_params = get_esd_trainable_parameters(esd_unet, args.train_method)

    # Set up optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

    # L2 Loss
    criteria = torch.nn.MSELoss()

    # Setup continual learning regularizers
    grad_hooks = setup_regularizers(args, pipe, esd_unet)

    # Cache the text embeddings that we'll re-use to reduce redundant computation
    conditioning_cache = prepare_conditioning_cache(pipe, args, torch_dtype)

    # Log training parameters
    print_training_summary(args)

    # Start the unlearning process
    try:
        _run_training_loop(
            args=args,
            pipe=pipe,
            base_unet=base_unet,
            esd_unet=esd_unet,
            optimizer=optimizer,
            criteria=criteria,
            conditioning_cache=conditioning_cache,
        )
    finally:
        cleanup_hooks(grad_hooks)

    # Save unlearned model
    save_esd_checkpoint(trainable_names, trainable_params, args.final_save_path)


def _run_training_loop(args, pipe, base_unet, esd_unet, optimizer, criteria, conditioning_cache):
    """Run optimization and optional simultaneous early stopping."""
    # Boolean flag whether to terminate training
    stop_training = False

    # Our current optimizer step
    iteration = 0

    # Where to store our temporary generated image for the Simultaneous unlearning setting
    eval_prompt_dir = os.path.abspath(os.path.expanduser(args.eval_prompt_dir)) if args.eval_prompt_dir else None

    # Start the training loop
    with tqdm(total=args.iterations, desc="Training ESD-SDXL", unit="iteration") as progress_bar:
        while not stop_training:
            # Clear optimizer
            optimizer.zero_grad()

            # Compute ESD unlearning loss
            esd_loss, logs = run_esd_training_step(
                pipe=pipe,
                base_unet=base_unet,
                esd_unet=esd_unet,
                cache=conditioning_cache,
                args=args,
                criteria=criteria,
            )

            # Track losses
            loss = esd_loss
            l1sp_loss = None
            l2sp_loss = None

            # Compute Weight Regularizers if selected
            if args.l1sp_weight > 0.0:
                from Regularizers.Weight.l1sp import calculate_l1sp_loss

                l1sp_loss = args.l1sp_weight * calculate_l1sp_loss(esd_unet, args.original_params)
                loss = loss + l1sp_loss
            if args.l2sp_weight > 0.0:
                from Regularizers.Weight.l2sp import calculate_l2sp_loss

                l2sp_loss = args.l2sp_weight * calculate_l2sp_loss(esd_unet, args.original_params)
                loss = loss + l2sp_loss

            # Compute gradient
            loss.backward()

            # Apply gradient projection if selected
            if args.with_gradient_projection and getattr(args, "aux_orth_proj", None) is not None:
                apply_gradient_projection(
                    unet=esd_unet,
                    aux_orth_proj=args.aux_orth_proj,
                    device=args.device,
                )

            # Apply gradient step
            optimizer.step()

            # Update trackers
            iteration += 1
            progress_bar.update(1)
            logs.update({"total_loss": loss.detach().item(), "esd_loss": esd_loss.detach().item()})
            if l1sp_loss is not None:
                logs["l1sp_loss"] = l1sp_loss.detach().item()
            if l2sp_loss is not None:
                logs["l2sp_loss"] = l2sp_loss.detach().item()
            progress_bar.set_postfix(**logs)

            # For Simultaneous setting, check if we should run unlearning evaluation
            if _should_evaluate(args, iteration):
                from Regularizers.Simultaneous.simultaneous import check_early_stopping, sample_and_evaluate_ua

                # Load UNet being unlearned and set to evaluation mode
                pipe.unet = esd_unet
                esd_unet.eval()

                # Get current Unlearning Accuracy
                ua = sample_and_evaluate_ua(
                    pipe,
                    args.concept_type,
                    iteration,
                    args.final_save_path,
                    args.target_concepts,
                    args.device,
                    args.eval_classifier_dir,
                    eval_prompt_dir,
                )

                # Restore Unlearning UNet to training mode
                esd_unet.train()

                # Log evaluation scores
                print(f"Iteration '{iteration}', Unlearned Accuracy: '{ua}'")
                args.best_ua, args.no_improvement_count, args.early_stop_triggered = check_early_stopping(
                    ua,
                    args.best_ua,
                    args.no_improvement_count,
                    args.eval_interval,
                    args.patience,
                    args.stop_threshold,
                )
                stop_training = args.early_stop_triggered

            # If we exceed our iteration limit
            if iteration >= args.iterations:
                stop_training = True


def _should_evaluate(args, iteration):
    return (
        args.eval_interval is not None
        and iteration >= args.eval_start
        and iteration % args.eval_interval == 0
    )


if __name__ == "__main__":
    main(parse_args())
