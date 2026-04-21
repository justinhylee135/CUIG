# Standard Library Imports
import os
import random
from pathlib import Path
import hashlib
import math
from PIL import Image
import ast

# Third Party Imports
import numpy as np
import openai
import regex as re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
import transformers
from transformers import get_scheduler


def anchor_dataset_collate_fn(examples, with_anchor_preservation):

    # Store target prompt tokens
    input_target_prompt_ids = [example["instance_target_prompt_ids"] for example in examples]

    # Store anchor prompt tokens
    input_anchor_prompt_ids = [example["instance_anchor_prompt_ids"] for example in examples]

    # Anchor image pixel values
    input_anchor_images = [example["instance_anchor_images"] for example in examples]

    # Masks for anchor images (defines valid regions)
    input_anchor_image_masks = [example["anchor_image_mask"] for example in examples]

    # Store anchor image file path
    input_anchor_image_paths = [example["instance_anchor_image_path"] for example in examples]

    # Store target prompt
    input_target_prompts = [example["instance_target_prompt"] for example in examples]
    input_anchor_prompts = [example["instance_anchor_prompt"] for example in examples]
    anchor_concepts = [example["anchor_concept"] for example in examples]

    # Concatenate preservation anchor examples and target examples so we can get output for both in one forward pass
    if with_anchor_preservation:
        # Append tokens
        input_target_prompt_ids += [example["preservation_anchor_prompt_ids"] for example in examples]

        # Append images
        input_anchor_images += [example["preservation_anchor_images"] for example in examples]

        # Append masks
        input_anchor_image_masks += [example["preservation_anchor_image_mask"] for example in examples]

    # Stack tokens vertically into a batch
    input_target_prompt_ids = torch.cat(input_target_prompt_ids, dim=0)
    input_anchor_prompt_ids = torch.cat(input_anchor_prompt_ids, dim=0)

    # Batch anchor images
    input_anchor_images = torch.stack(input_anchor_images)
    input_anchor_images = input_anchor_images.to(memory_format=torch.contiguous_format).float()

    # Batch anchor image masks
    input_anchor_image_masks = torch.stack(input_anchor_image_masks)
    input_anchor_image_masks = input_anchor_image_masks.to(memory_format=torch.contiguous_format).float()

    # Create anchor dataloader batch
    batch = {
        "input_target_prompt_ids": input_target_prompt_ids,
        "input_anchor_prompt_ids": input_anchor_prompt_ids,
        "input_anchor_images": input_anchor_images,
        "input_anchor_image_masks": input_anchor_image_masks.unsqueeze(1),
        "input_anchor_image_paths": input_anchor_image_paths,
        "input_target_prompts": input_target_prompts,
        "input_anchor_prompts": input_anchor_prompts,
        "anchor_concepts": anchor_concepts
    }

    # Return batch
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the anchor prompts to generate anchor images on multiple GPUs."

    def __init__(self, anchor_prompt_list, num_to_generate):
        # Initialize prompt list and number of prompts
        self.anchor_prompt_list_length = len(anchor_prompt_list)
        self.anchor_prompt_index_offset = 0

        if num_to_generate < self.anchor_prompt_list_length:
            print(
                f"Storing last '{num_to_generate}' prompts of "
                f"'{self.anchor_prompt_list_length}' in PromptDataset"
            )
            self.anchor_prompt_index_offset = self.anchor_prompt_list_length - num_to_generate
            anchor_prompt_list = anchor_prompt_list[-num_to_generate:]

        self.anchor_prompt_list = anchor_prompt_list
        self.num_to_generate = num_to_generate

    def __len__(self):
        # Return number of images we're going to generate
        return self.num_to_generate

    def __getitem__(self, index):
        # We will return a map
        example = {}

        # Store prompt and index in map
        # If num_to_generate > len(anchor_prompt_list) we'll just have to wrap around the anchor_prompt list
        stored_index = index % len(self.anchor_prompt_list)
        original_index = (stored_index + self.anchor_prompt_index_offset) % self.anchor_prompt_list_length

        example["anchor_prompt_index"] = original_index
        example["anchor_prompt"] = self.anchor_prompt_list[stored_index]

        return example


class AnchorDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concept_configs,
        concept_type,
        with_style_replacement,
        tokenizer,
        resolution=512,
        center_crop=False,
        with_anchor_preservation=False,
        num_anchor_images=200,
        hflip=False,
        aug=True,
    ):
        # Image Settings
        self.resolution = resolution
        self.center_crop = center_crop
        self.interpolation = Image.LANCZOS
        self.aug = aug

        # Store tokenizer
        self.tokenizer = tokenizer

        # Store concept type (ex. style or object)
        self.concept_type = concept_type
        self.with_style_replacement = with_style_replacement

        # List of tuples for 
        self.anchor_target_datasets = []
        self.anchor_datasets = []
        self.with_anchor_preservation = with_anchor_preservation

        # Iterate through each concept configuration
        for concept in concept_configs:
            # Load anchor image file paths 
            with open(concept["instance_anchor_image_paths_file"], "r") as f:
                inst_anchor_image_paths = f.read().splitlines()

            # Load anchor prompts file
            with open(concept["instance_anchor_prompts_file"], "r") as f:
                inst_anchor_prompts = f.read().splitlines()

            # Create tuple of (image, prompt, anchor_target_concept)
            inst_anchor_target_dataset = [
                (image, prompt, concept["anchor_target_concept"])
                for (image, prompt) in zip(inst_anchor_image_paths, inst_anchor_prompts)
            ]

            # Append to our anchor target datasets
            self.anchor_target_datasets.extend(inst_anchor_target_dataset)

            # Create dataset with just anchor image and prompt
            if with_anchor_preservation:
                inst_anchor_dataset = [
                    (image, prompt) for (image, prompt) in zip(inst_anchor_image_paths, inst_anchor_prompts)
                ]
                self.anchor_datasets.extend(inst_anchor_dataset[:num_anchor_images])

        # Shuffle dataset, so we unlearn all the concepts in a randomized batch
        random.shuffle(self.anchor_target_datasets)

        # Number of images in our anchor target datasets
        self.num_anchor_target_images = len(self.anchor_target_datasets)

        # Number of images in our anchor dataset
        self.num_anchor_images = len(self.anchor_datasets)

        # Whether to randomly horizontally flip anchor images
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        # Transformation applied to anchor images
        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        # Return size of anchor target dataset (anchor dataset is always smaller)
        return self.num_anchor_target_images

    def preprocess(self, image, scale, resample):
        """
        Preprocess an image by resizing, normalizing, and creating a mask.
        
        Args:
            image: PIL Image to preprocess
            scale: Target scale for resizing the image
            resample: Resampling filter for image resize
            
        Returns:
            tuple: (instance_image, mask) - preprocessed image array and attention mask
        """
        # Determine outer and inner dimensions for cropping
        # outer = larger dimension, inner = smaller dimension (resolution)
        outer, inner = self.resolution, scale
        if scale > self.resolution:
            outer, inner = scale, self.resolution
        
        # Generate random crop coordinates
        # Ensures crop is within bounds of outer dimension
        top = np.random.randint(0, outer - inner + 1)
        left = np.random.randint(0, outer - inner + 1)
        
        # Resize image to target scale and convert to normalized float array
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)

        # Normalize image from [0, 255] to [-1.0, 1.0] range
        image = (image / 127.5 - 1.0).astype(np.float32)
        
        # Initialize output tensors
        instance_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)

        # Mask is at 1/8 resolution (for VAE latent space compatibility)
        mask = np.zeros((self.resolution // 8, self.resolution // 8))
        
        # Handle two cases: upscaled or downscaled images
        if scale > self.resolution:
            # Image is larger than target resolution, so crop it
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.resolution // 8, self.resolution // 8))
        else:
            # Image is smaller than target resolution, pad it with zeros
            instance_image[top : top + inner, left : left + inner, :] = image
            # Create mask for valid region (with padding for VAE encoder)
            mask[
                top // 8 + 1 : (top + scale) // 8 - 1,
                left // 8 + 1 : (left + scale) // 8 - 1,
            ] = 1.0
        
        return instance_image, mask

    def __getprompt__(self, instance_anchor_prompt, instance_anchor_target_concept):
        # This is the anchor concept we will map the target concept to (for style this is optional)
        instance_anchor_concept=""

        # Style unlearning prompts
        if self.concept_type == "style":
            # Clean up instance_anchor_target_concept
            instance_anchor_target_concept = instance_anchor_target_concept.replace("_", " ")
            instance_anchor_target_concept = instance_anchor_target_concept.replace(" Style", "")

            # If selected, replace style in anchor prompt with target prompt directly
            if self.with_style_replacement:
                # Find the style in the anchor prompt and then replace the string with our target concept
                instance_target_prompt = re.sub(r"in\s+.*?style", f"in {instance_anchor_target_concept} style", instance_anchor_prompt, flags=re.IGNORECASE)

                # If we couldn't identify a style in the anchor prompt, then just hard code a replacement with our target concept
                if instance_target_prompt == instance_anchor_prompt:
                    instance_target_prompt = f"An image in {instance_anchor_target_concept} Style"
                    print(f"Unsuccessful replacement for style '{instance_anchor_target_concept}' in prompt '{instance_anchor_prompt}'. Using '{instance_target_prompt}' instead.")
            else:
                # Instead of string replacement just randomly prepend (0) or append (1) the target concept to the anchor prompt
                prompt_choice = np.random.choice([0, 1])

                # Prepend or append our target concept
                if prompt_choice == 0: # Prepend
                    instance_target_prompt = instance_anchor_prompt[0].lower() + instance_anchor_prompt[1:] # Make first character lowercase
                    instance_target_prompt = f"In {instance_anchor_target_concept} Style, {instance_target_prompt}"
                elif prompt_choice == 1: # Append
                    instance_target_prompt = instance_anchor_prompt.replace(".", "") # Remove period at the end
                    instance_target_prompt = f"{instance_target_prompt}, in {instance_anchor_target_concept} Style."

        elif self.concept_type in ["object", "celeb", "character", "nudity"]: # All other concept types require anchor and target
            # separate anchor and target concept
            instance_anchor_concept, instance_target_concept = instance_anchor_target_concept.split("+")

            # '*' is the overide character, this just replaces the entire anchor prompt with the target concept
            if instance_anchor_concept == "*":
                instance_target_prompt = instance_target_concept
            else:
                # Replace anchor concept with target in prompt via string replacement
                instance_target_prompt = re.sub(re.escape(instance_anchor_concept), instance_target_concept, instance_anchor_prompt, flags=re.IGNORECASE)
                
                # Ensure something changed
                if instance_target_prompt == instance_anchor_prompt:
                    # If string replacement couldn't find anchor concept in anchor prompt just replace the entire anchor prompt with target concept
                    instance_target_prompt = f"A {instance_target_concept} image"
                    print(f"Unsuccessful replacement for anchor concept '{instance_anchor_concept}' in prompt '{instance_anchor_prompt}'. Using '{instance_target_prompt}' instead.")
        
        # Return the target prompt (anchor prompt but modified with target concept), and the anchor concept (empty string for style)
        return instance_target_prompt, instance_anchor_concept

    def __getitem__(self, index):
        # The map we're going to return
        example = {}

        # Extract (image, prompt, anchor+target concept)
        instance_anchor_image, instance_anchor_prompt, instance_anchor_target_concept = self.anchor_target_datasets[index % self.num_anchor_target_images]

        # Store anchor image file path
        example["instance_anchor_image_path"] = instance_anchor_image

        # Open anchor image
        instance_anchor_image = Image.open(instance_anchor_image)

        # Convert anchor image to RGB
        if not instance_anchor_image.mode == "RGB": instance_anchor_image = instance_anchor_image.convert("RGB")

        # Apply horizontal flip if specified
        instance_anchor_image = self.flip(instance_anchor_image)

        # Acquire target prompt and the anchor concept
        instance_target_prompt, anchor_concept = self.__getprompt__(instance_anchor_prompt, instance_anchor_target_concept)

        # Store the anchor prompt, modified anchor prompt (target prompt), and then the anchor target (tht was used for string replacement)
        example["instance_target_prompt"] = instance_target_prompt
        example["instance_anchor_prompt"] = instance_anchor_prompt
        example["anchor_concept"] = anchor_concept

        # Apply image augmentations if specified
        random_scale = self.resolution
        if self.aug:
            random_scale = (
                np.random.randint(self.resolution // 3, self.resolution + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.resolution), int(1.4 * self.resolution))
            )
        instance_anchor_image, mask = self.preprocess(
            instance_anchor_image, random_scale, self.interpolation
        )
        if random_scale < 0.6 * self.resolution:
            instance_target_prompt = (
                np.random.choice(["A far away ", "A very small "]) + instance_target_prompt
            )
        elif random_scale > self.resolution:
            instance_target_prompt = (
                np.random.choice(["A zoomed in ", "A close up "]) + instance_target_prompt
            )

        # Store anchor image tensor and the mask (valid region)
        example["instance_anchor_images"] = torch.from_numpy(instance_anchor_image).permute(2, 0, 1)
        example["anchor_image_mask"] = torch.from_numpy(mask)

        # Tokenize the instance target prompt (modified anchor prompt)
        example["instance_target_prompt_ids"] = self.tokenizer(
            instance_target_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # Tokenize the original anchor promtp
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # If specified, we're going to apply standard diffusion training loss to the anchor concept
        if self.with_anchor_preservation:
            # Extract anchor image (file path) and anchor prompt to preserve
            preservation_anchor_image, preservation_anchor_prompt = self.anchor_datasets[index % self.num_anchor_images]

            # Open anchor image file
            preservation_anchor_image = Image.open(preservation_anchor_image)

            # Convert to anchor image to RGB
            if not preservation_anchor_image.mode == "RGB": preservation_anchor_image = preservation_anchor_image.convert("RGB")

            # Apply transformation to anchor image
            example["preservation_anchor_images"] = self.image_transforms(preservation_anchor_image)

            # All pixels are valid regions because we did not crop
            example["preservation_anchor_image_mask"] = torch.ones_like(example["anchor_image_mask"])

            # Tokenize the anchor prompt
            example["preservation_anchor_prompt_ids"] = self.tokenizer(
                preservation_anchor_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def generate_anchor_images_if_needed(args, accelerator, logger):
    """
    Generate anchor images if prior preservation is enabled and images don't exist.

    Args:
        args: Arguments object containing configuration parameters
        accelerator: Accelerator object for distributed training
        logger: Logger for info messages
        
    Returns:
        None: Modifies args.concept_configs in place
    """
    # Build string of concepts to exclude from prompt generation
    # We don't want to accidentally include a concept we're going to erase into our anchor dataset
    excluded_concepts = []
    for i, concept in enumerate(args.concept_configs):
        # String replacement in case we're unlearning styles
        # For styles we appended " Style" in utils/set_concepts_list() but let's ignore that
        excluded_concepts.append(
            concept["anchor_target_concept"].replace("_", " ").replace(" Style", "")
        )

    # If the user wants to pass the concepts previously unlearned as well, then include those too in the excluded list
    if args.previously_unlearned:
        excluded_concepts.extend(
            ast.literal_eval(args.previously_unlearned)
        )

    # Turn the excluded list into one big string separated by commas    
    if len(excluded_concepts) > 1:
        excluded_concepts_str = ", ".join(excluded_concepts[:-1]) + ", or " + excluded_concepts[-1]
    else:
        excluded_concepts_str = ", ".join(excluded_concepts)

    # Iterate through each concept configuration
    print(f"Processing '{len(args.concept_configs)}' concepts.")
    for i, concept_cfg in enumerate(args.concept_configs):
        # If the user already passed 'concept_configs' because they wanted to write it themselves
        # Then just use the the path to the anchor prompts and image paths given
        if (
            concept_cfg["instance_anchor_prompts_file"] is not None
            and concept_cfg["instance_anchor_image_paths_file"] is not None
        ):
            break
        
        # This folder will hold our anchor dataset
        anchor_dataset_dir = Path(concept_cfg["anchor_dataset_dir"])
        if not anchor_dataset_dir.exists():
            anchor_dataset_dir.mkdir(parents=True, exist_ok=True)

        # This subfolder will hold the anchor images
        image_dir = (Path(os.path.join(anchor_dataset_dir, "images")))
        os.makedirs(image_dir, exist_ok=True)
        
        # Time to generate the anchor images, but make sure they don't already exist
        num_existing_class_images = len(list(image_dir.iterdir()))
        print(f"\t{i+1}. '{concept_cfg['anchor_target_concept']}': '{num_existing_class_images}' anchor images already found at '{image_dir}'.")
        if (
            num_existing_class_images < args.num_anchor_images
        ):  
            # Define number of concepts to generate
            num_to_generate = args.num_anchor_images - num_existing_class_images
            print(f"Only '{num_existing_class_images}' class images found at '{image_dir}'. Generating '{num_to_generate}' more images...")

            # Set up diffusion model pipeline
            diffusion_pipeline = _setup_generation_pipeline(args, accelerator)

            # Collect anchor prompts in list (Generate if needed)
            anchor_prompt_list = _get_anchor_prompts_list( 
                args, concept_cfg, excluded_concepts_str, anchor_dataset_dir, diffusion_pipeline, accelerator
            )

            # Generate and save anchor images
            _generate_images_from_prompts(args, num_to_generate, anchor_prompt_list, anchor_dataset_dir, diffusion_pipeline, accelerator)

            # Remove diffusion pipeline from memory
            del diffusion_pipeline
        else:
            # Just print message that we're skipping anchor image generation
            print(f"Skipping anchor image generation because already found '{num_existing_class_images}' images in '{image_dir}'")

        # Update concept config with new paths
        _update_concept_config_file_paths(concept_cfg, anchor_dataset_dir)

        # Clear GPU memory just in case
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Completed initializing anchor datasets for '{len(args.concept_configs)}' concepts")

def _setup_generation_pipeline(args, accelerator):
    """Set up the diffusion pipeline for anchor dataset image generation."""
    torch_dtype = (
        torch.float16 if accelerator.device.type == "cuda" else torch.float32
    )

    # Default is fp16, we shouldn't really need full precision for inference
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16

    # Use diffusers pipeline    
    pipeline = DiffusionPipeline.from_pretrained(
        args.base_model_dir, # Our diffusion model (we haven't unlearned anything yet)
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.hf_revision,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)
    
    return pipeline


def _get_anchor_prompts_list(args, concept_cfg, excluded_concepts_str, anchor_dataset_dir, diffusion_pipeline, accelerator):
    """Get collection of prompts for class image generation."""

    # Generate anchor prompts if file doesn't exist
    if not os.path.isfile(concept_cfg["anchor_prompt_path"]):
        # Warning Message
        print(f"WARNING: '{concept_cfg['anchor_prompt_path']}' is not a path to a text file with anchor prompts. Will instead generate them...")

        # Get the unlearning anchor+target pair
        anchor_target_concept = concept_cfg['anchor_target_concept']
        print(f"Generating anchor prompts for '{anchor_target_concept}' and saving to '{concept_cfg['anchor_prompt_path']}'")

        # Generate anchor prompts
        anchor_prompt_list = _generate_anchor_prompts(
            diffusion_pipeline,
            anchor_target_concept, 
            excluded_concepts_str,
            args.concept_type,
            args.num_anchor_prompts, # Number of anchor promtps to generate (default: 200)
            llm_model_id=args.prompt_gen_model
        )

        # Save generated anchor prompts to txt file
        with open(concept_cfg["anchor_prompt_path"], "w") as f:
            for prompt in anchor_prompt_list:
                f.write(prompt.strip() + "\n")
        print(f"Saved {len(anchor_prompt_list)} anchor prompts to '{concept_cfg['anchor_prompt_path']}'.")
    else:
        # Load in anchor prompts if file path exists
        print(f"Loading anchor prompts from '{concept_cfg['anchor_prompt_path']}' for anchor_target '{concept_cfg['anchor_target_concept']}'")
        with open(concept_cfg["anchor_prompt_path"]) as f:
            anchor_prompt_list = [x.strip() for x in f.readlines() if x.strip()]

        # If prompt file exists but has too few prompts, generate the remainder.
        if len(anchor_prompt_list) < args.num_anchor_prompts:

            # Determine number of additional anchor prompts to generate
            num_remaining_prompts = args.num_anchor_prompts - len(anchor_prompt_list)
            anchor_target_concept = concept_cfg["anchor_target_concept"]
            print(
                f"Only '{len(anchor_prompt_list)}' anchor prompts found in "
                f"'{concept_cfg['anchor_prompt_path']}'. Generating '{num_remaining_prompts}' more..."
            )

            # Store additionally generated anchor prompts
            additional_anchor_prompt_list, _ = _generate_anchor_prompts(
                diffusion_pipeline,
                anchor_target_concept,
                excluded_concepts_str,
                args.concept_type,
                num_remaining_prompts,
                llm_model_id=args.prompt_gen_model,
            )
            additional_anchor_prompt_list = [x.strip() for x in additional_anchor_prompt_list if x.strip()]
            anchor_prompt_list.extend(additional_anchor_prompt_list)

            # Update txt file with current and new anchor prompts
            with open(concept_cfg["anchor_prompt_path"], "w") as f:
                for prompt in anchor_prompt_list:
                    f.write(prompt.strip() + "\n")
            print(
                f"Updated '{concept_cfg['anchor_prompt_path']}' with "
                f"'{len(anchor_prompt_list)}' total anchor prompts."
            )
    
    # Return generated or retrieved anchor prompt list
    return anchor_prompt_list


def _generate_images_from_prompts(args, num_to_generate, anchor_prompt_list, 
                                anchor_dataset_dir, diffusion_pipeline, accelerator):
    """Generate images from the prompt collection."""

    # Create anchor prompt dataset and dataloader
    print(f"Number of anchor images to sample: '{num_to_generate}'.")
    anchor_prompt_dataset = PromptDataset(anchor_prompt_list, num_to_generate)
    anchor_prompt_dataloader = torch.utils.data.DataLoader(
        anchor_prompt_dataset, batch_size=args.sample_batch_size
    )
    anchor_prompt_dataloader = accelerator.prepare(anchor_prompt_dataloader)

    # Clean up anchor dataset files (in case of errors)
    if os.path.exists(f"{anchor_dataset_dir}/prompts.txt"):
        print(f"Rebuiliding existing prompts file: '{anchor_dataset_dir}/prompts.txt'")
        os.remove(f"{anchor_dataset_dir}/prompts.txt")
    if os.path.exists(f"{anchor_dataset_dir}/images.txt"):
        print(f"Rebuiliding existing images file: '{anchor_dataset_dir}/images.txt'")
        os.remove(f"{anchor_dataset_dir}/images.txt")

    # Rebuild anchor dataset
    anchor_images_dir = anchor_dataset_dir / "images"
    if os.path.isdir(anchor_images_dir):
        # We're going to rebuild the list of image paths and prompts to write to the txt file
        rebuilt_image_paths = []
        rebuilt_prompts = []

        # Get each image
        with os.scandir(anchor_images_dir) as image_entries:
            for image_entry in image_entries:
                if not image_entry.is_file():
                    continue

                # Extract the anchor prompt id from the image file name
                anchor_prompt_id = int(image_entry.name.split("-", 1)[0])

                # Append image path and anchor prompt
                rebuilt_image_paths.append(image_entry.path)
                rebuilt_prompts.append(anchor_prompt_list[anchor_prompt_id - 1])

        # Write image path and prompts to txt files
        if rebuilt_image_paths:
            with open(f"{anchor_dataset_dir}/images.txt", "a") as images_file_writer:
                images_file_writer.write("\n".join(rebuilt_image_paths) + "\n")
            with open(f"{anchor_dataset_dir}/prompts.txt", "a") as prompts_file_writer:
                prompts_file_writer.write("\n".join(rebuilt_prompts) + "\n")

    # Generate anchor dataset images
    for example in tqdm(
        anchor_prompt_dataloader,
        desc="Generating anchor dataset images",
        disable=not accelerator.is_local_main_process,
    ):
        # Ensure GPUs are in sync
        accelerator.wait_for_everyone()

        # Initialize file writers for writing the paths for the images and the corresponding prompt
        with open(f"{anchor_dataset_dir}/prompts.txt", "a") as prompts_file_writer, open(f"{anchor_dataset_dir}/images.txt", "a") as images_file_writer:
            # Generate image
            images = diffusion_pipeline(
                example["anchor_prompt"],
                num_inference_steps=100,
                guidance_scale=6.0,
                negative_prompt = None,
                eta=1.0,
                height = args.resolution, # Default: 512
                width = args.resolution, # Default: 512
            ).images

            # Iterate through generated images batch
            for i, image in enumerate(images):
                # Compute hash for file name
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = (
                    anchor_images_dir
                    / f"{example['anchor_prompt_index'][i]+1}-{hash_image}.jpg"
                )

                # Save image
                image.save(image_filename)

                # Write image path to images.txt
                images_file_writer.write(str(image_filename) + "\n")

            # Write corresponding anchor prompt to prompts.txt
            prompts_file_writer.write("\n".join(example["anchor_prompt"]) + "\n")

            # Ensure GPUs are in sync
            accelerator.wait_for_everyone()
    
    print(f"Finished generating '{num_to_generate}' images")


def _update_concept_config_file_paths(concept, anchor_dataset_dir):
    """Update concept config with anchor dataset file paths."""
    concept["instance_anchor_prompts_file"] = os.path.join(anchor_dataset_dir, "prompts.txt")
    concept["instance_anchor_image_paths_file"] = os.path.join(anchor_dataset_dir, "images.txt")
    
def _generate_anchor_prompts(
    diffusion_pipeline,
    anchor_target_concept,
    excluded_concepts_str,
    concept_type,
    num_anchor_prompts=200,
    llm_model_id="openai",
):
    """Generate anchor prompts to use for generating anchor images for anchor dataset"""

    # Get OpenAI API key if using openai model
    if llm_model_id == "openai": # Default is openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        llm_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = transformers.pipeline(
            "text-generation",
            model=llm_model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    # We will store our anchor prompts here
    anchor_prompt_list = []

    # Build LLM prompt below
    if concept_type == "style":    # Style Unlearning
        # Note to whoever is reading this code: The actual prompts used for style unlearning is taken straight for ConAbl's repo. They are just captions from LAION dataset
        # This is a back-up, if you would rather generate prompts yourself. But, for reproducibility just use the given prompts in this repo.
        llm_prompt = [
            {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line."},
            {"role": "user", "content": f"Generate {num_anchor_prompts} random prompts in the format 'A {{object}} image in {{style}} style'. Don't include any prompts with styles similar to {excluded_concepts_str}."},
        ]
    elif concept_type in ["object", "celeb", "character"]:
        # Extract anchor concept from anchor_target pair
        anchor_concept = anchor_target_concept.split("+")[0]

        # Prompt for object unleraning (works for pretty much anything else that has a specific anchor concept)
        llm_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert at generating concise, descriptive image captions for synthetic image generation. "
                    "Your output is a list of short captions, one per line, with no commentary or explanation. "
                    "Each caption should describe an image containing exactly one subject."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Generate {num_anchor_prompts} short captions for images that feature exactly one {anchor_concept}. "
                    f"The image must not contain any other animals, humans, or objects. "
                    f"Each caption must include the word \"{anchor_concept}\". "
                    f"The caption should be general enough that replacing the word \"{anchor_concept}\" with another animal, human, or object would still result in a plausible description."
                )
            }
        ]
    print(f"Using GPT Prompt: '{llm_prompt[1]['content']}'")

    # Keep prompting LLM until we get the number of prompts that we need
    num_llm_calls = 1
    while True:
        # API Call for gpt
        if llm_model_id == "openai":
            llm_output = openai.ChatCompletion.create(
                model="gpt-4.1", messages=llm_prompt
            ).choices[0].message.content.lower().split("\n")
        else: # Prompt LLM Locally
            # Define tokens to stop LLM generation
            terminators = [
                diffusion_pipeline.tokenizer.eos_token_id,
                diffusion_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Prompt local LLM
            llm_output = model(
                llm_prompt,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )[0]["generated_text"][-1]['content'].split("\n")[1:-1]

        # Print number of prompts generated for this iteration
        print(f"{num_llm_calls}: '{len(llm_output)}' prompts generated.")
        
        # Store prompts into list
        if concept_type == "style":
            # For style, just make sure the generated prompts are not empty strings
            anchor_prompt_list += [
                x for x in llm_output if x != ''
            ]
        else:
            # For anything else (requires anchor prompt), make sure prompt isn't empty AND that it contains the anchor concept
            anchor_prompt_list += [
                x
                for x in llm_output
                if (anchor_concept in x and x != '')
            ]

        # Update the LLM Prompt with this iteration's LLM output so it nows not to generate duplicates
        llm_prompt.append(
            {"role": "assistant", "content": "\n".join(llm_output)}
        )

        # Ask to generate remaining amount until we get the number of prompts that we need
        llm_prompt.append(
            {
                "role": "user",
                "content": f"Generate {num_anchor_prompts-len(anchor_prompt_list)} more captions",
            }
        )
        
        # Keep last 10 elements to make sure prompt doesn't grow too long
        llm_prompt = llm_prompt[-10:]

        # Update iteration counter
        num_llm_calls +=1

        # Stop generating once we reached the number of prompts we need or if we tried more than 10 times
        if len(anchor_prompt_list) >= num_anchor_prompts or num_llm_calls > 10:
            break
    
    # Clean up our generated anchor prompts
    anchor_prompt_list = _clean_up_anchor_prompt_list(anchor_prompt_list)[:num_anchor_prompts]  

    # Return generated anchor prompts
    return anchor_prompt_list


def _clean_up_anchor_prompt_list(anchor_prompt_list):
    """Clean up anchor prompts to remove formatting errors"""

    # Remove all digits from prompts
    anchor_prompt_list = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in anchor_prompt_list
    ]
    
    # Remove leading periods from prompts
    anchor_prompt_list = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in anchor_prompt_list
    ]

    # Strip leading and trailing whitespace from each prompt
    anchor_prompt_list = [x.strip() for x in anchor_prompt_list]

    # Remove all double quotes from prompts
    anchor_prompt_list = [x.replace('"', "") for x in anchor_prompt_list]

    # Return cleaned up anchor prompt list
    return anchor_prompt_list

def setup_data_and_scheduler(args, tokenizer, optimizer):
    """
    Set up dataset, dataloader, and learning rate scheduler.
    
    Args:
        args: Arguments object
        tokenizer: Tokenizer
        optimizer: Optimizer
        
    Returns:
        tuple: (anchor_dataloader, lr_scheduler) and modifies args with calculated steps
    """
    # Create dataset and dataloader
    num_anchor_images = args.num_anchor_images
    print(f"\tCreating anchor dataset for '{len(args.concept_configs)}' concepts with '{num_anchor_images}' images each ('{len(args.concept_configs) * num_anchor_images}' total)...")

    # Initialize anchor dataset
    anchor_dataset = AnchorDataset(
        concept_configs=args.concept_configs,
        concept_type=args.concept_type,
        with_style_replacement=args.with_style_replacement, # Default: False
        tokenizer=tokenizer,
        with_anchor_preservation=args.with_anchor_preservation, # Default: False
        resolution=args.resolution, # Default: 512
        center_crop=args.center_crop, # Default: False
        num_anchor_images=num_anchor_images,
        hflip=args.hflip, # I personally use this
        aug=not args.noaug, # I personally use this (don't augment)
    )

    # Create the dataloader
    print(f"\tCreating anchor dataloader with '{len(anchor_dataset)}' examples and batchsize '{args.anchor_batch_size}'...")
    anchor_dataloader = torch.utils.data.DataLoader(
        anchor_dataset,
        batch_size=args.anchor_batch_size, # Default: 4
        shuffle=True,
        collate_fn=lambda examples: anchor_dataset_collate_fn(examples, args.with_anchor_preservation),
        num_workers=args.dataloader_num_workers # Default: 2
    )

    # Store whether iterations was not given and then automatically calculated
    overrode_iterations = False

    # (default gradient acc. steps is 1)
    num_iterations_per_epoch = math.ceil(len(anchor_dataloader) / args.gradient_accumulation_steps)

    # Automatically calculcate iterations
    if args.iterations is None:
        # Default number of epochs is 0
        args.iterations = args.epochs * num_iterations_per_epoch
        overrode_iterations = True

    # Print number of training steps
    print(f"\tNumber of training steps: '{args.iterations}'")

    # Create learning rate scheduler (default: constant)
    print(f"\tUsing lr scheduler: '{args.lr_scheduler}'")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, # Default (500 * 1 = 500)
        num_training_steps=args.iterations * args.gradient_accumulation_steps, 
    )

    # Store for later use
    args._overrode_iterations = overrode_iterations

    # Return anchor dataloader and lr scheduler
    return anchor_dataloader, lr_scheduler, anchor_dataset
