# Standard Library Imports
import os
import random
import shutil
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
import transformers
from transformers import get_scheduler

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    image_paths = [example["image_path"] for example in examples]
    target_prompts = [example["target_prompt"] for example in examples]
    target_concepts = [example["target_concept"] for example in examples]
    
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1),
        "image_paths": image_paths,
        "target_prompts": target_prompts,
        "target_concepts": target_concepts,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt[index % len(self.prompt)]
        example["index"] = index
        return example


class ForgetMeNotDataset(Dataset):
    """
    A dataset to prepare the instance and class images with prompts for Forget Me Not training.
    It pre-processes the images and tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        concept_type,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.LANCZOS
        self.aug = aug
        self.concept_type = concept_type

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        
        for concept in concepts_list:
            # Load instance images and prompts (target concept to forget)
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            
            inst_img_path = [
                (x, y, concept["caption_target"])
                for (x, y) in zip(inst_images_path, inst_prompt)
            ]
            self.instance_images_path.extend(inst_img_path)

            # Load class images for regularization if enabled
            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [
                        concept["class_prompt"] for _ in range(len(class_images_path))
                    ]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [
                    (x, y) for (x, y) in zip(class_images_path, class_prompt)
                ]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(
            0, outer - inner + 1
        )
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // 8 + 1 : (top + scale) // 8 - 1,
                left // 8 + 1 : (left + scale) // 8 - 1,
            ] = 1.0
        return instance_image, mask

    def __getprompt__(self, instance_prompt, instance_target):
        """
        Generate prompts for FMN unlearning containing the target concept to forget.
        The goal is to create prompts that prominently feature the concept we want to unlearn.
        """
        return instance_prompt

        if self.concept_type == "style":
            # Clean up instance_target
            instance_target = instance_target.replace("_", " ")
            instance_target = instance_target.replace(" Style", "")
            
            # For FMN unlearning, we want prompts with the target style prominently featured
            r = np.random.choice([0, 1, 2, 3])
            instance_prompt = instance_prompt.replace(".", "")
            if r == 0:
                instance_prompt = f"{instance_prompt}, in {instance_target} style"
            elif r == 1:
                instance_prompt = f"In {instance_target} style, {instance_prompt}"
            elif r == 2:
                instance_prompt = f"{instance_prompt} in the style of {instance_target}"
            else:
                instance_prompt = f"A {instance_target} style {instance_prompt}"
                
        elif self.concept_type == "object":
            # For FMN with objects, prominently feature the target object
            if "+" in instance_target:
                # Format: "anchor+target" - we use target for unlearning
                _, target = instance_target.split("+")
            else:
                target = instance_target
            
            # Generate varied prompts featuring the target object
            r = np.random.choice([0, 1, 2, 3])
            if r == 0:
                instance_prompt = f"a photo of {target}"
            elif r == 1:
                instance_prompt = f"a {target}"
            elif r == 2:
                instance_prompt = f"{target} in the image"
            else:
                instance_prompt = f"an image featuring {target}"
                
        elif self.concept_type in ["nudity", "inappropriate_content"]:
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_target}, {instance_prompt}"
                if r == 0
                else f"in {instance_target} style, {instance_prompt}"
                if r == 1
                else f"{instance_prompt}, {instance_target}"
            )
        elif self.concept_type == "memorization":
            if "+" in instance_target:
                instance_prompt = instance_target.split("+")[1]
            else:
                instance_prompt = instance_target
                
        return instance_prompt

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt, instance_target = self.instance_images_path[
            index % self.num_instance_images
        ]
        example["image_path"] = instance_image
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        
        # Handle multiple concepts if semicolon-separated
        if ";" in instance_target:
            instance_target = instance_target.split(";")
            instance_target = instance_target[index % len(instance_target)]

        # Generate the prompt with target concept
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        example["target_prompt"] = instance_prompt
        
        # Extract target concept
        if self.concept_type == "style":
            target_concept = instance_target.replace("_", " ")
            target_concept = target_concept.replace(" Style", "")
        elif self.concept_type == "object":
            if "+" in instance_target:
                _, target_concept = instance_target.split("+")
            else:
                target_concept = instance_target
        example["target_concept"] = target_concept


        # Apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(
            instance_image, random_scale, self.interpolation
        )

        # Add zoom-related prefixes if needed
        if random_scale < 0.6 * self.size:
            instance_prompt = (
                np.random.choice(["a far away ", "very small "]) + instance_prompt
            )
        elif random_scale > self.size:
            instance_prompt = (
                np.random.choice(["zoomed in ", "close up "]) + instance_prompt
            )

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # Add class/regularization images if using prior preservation
        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[
                index % self.num_class_images
            ]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def generate_unlearning_images(args, accelerator, logger):
    """
    Generate unlearning images that contain the target concept to forget.
    This is for FMN training where we need images with the concept we want to unlearn.
    """
        
    print(f"Processing '{len(args.concepts_list)}' concepts for unlearning image generation.")

    # For FMN, we want to generate images WITH the target concepts for unlearning
    target_concepts = []
    for concept in args.concepts_list:
        target_concepts.append(
            concept["caption_target"].replace("_", " ").replace(" Style", "")
        )

    for i, concept in enumerate(args.concepts_list):
        # Check if paths are already provided
        if concept["instance_prompt"] is not None and concept["instance_data_dir"] is not None:
            break
        class_images_dir = Path(concept["class_data_dir"])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(f"{class_images_dir}/images", exist_ok=True)
        
        # Check existing images
        image_dir = Path(os.path.join(class_images_dir, "images"))
        num_existing_class_images = len(list(image_dir.iterdir()))
        print(f"\t{i+1}. '{concept['caption_target']}': '{num_existing_class_images}' unlearning images at '{image_dir}'.")
        
        if num_existing_class_images < args.num_class_images:
            num_to_generate = args.num_class_images - num_existing_class_images
            print(f"Generating '{num_to_generate}' more unlearning images...")
            
            # Generate unlearning prompts containing the target concept
            pipeline = _setup_generation_pipeline(args, accelerator)
            class_prompt_collection = _get_unlearning_prompts(
                args, concept, target_concepts[i], class_images_dir, 
                args.num_class_prompts, args.prompt_gen_model
            )
            
            # Generate images from prompts
            _generate_images_from_prompts(
                args, num_to_generate, concept, class_prompt_collection, 
                class_images_dir, pipeline, accelerator, logger
            )
            del pipeline

        # Update concept paths
        _update_concept_paths(concept, class_images_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_unlearning_prompts(args, concept, target_concepts_str, class_images_dir, 
                           num_prompts=200, model_id="meta-llama"):
    """
    Generate diverse prompts for unlearning that contain the target concept.
    """
    if not os.path.isfile(concept["class_prompt"]):
        print(f"Generating unlearning prompts containing '{target_concepts_str}' using '{model_id}' model...")
        
        if model_id == "simple":
            # Simple format without GPT calls
            target_concept = concept["caption_target"].replace("_", " ").replace(" Style", "")
            
            if args.concept_type == "object":
                outputs = [f"a photo of {target_concept}"] * num_prompts
            elif args.concept_type == "style":
                outputs = [f"a photo in the style of {target_concept}"] * num_prompts
            else:
                # Fallback for other types
                outputs = [f"a photo of {target_concept}"] * num_prompts
                
        elif model_id == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            messages = [
                {
                    "role": "system", 
                    "content": "You generate diverse image captions containing specific concepts for unlearning. Output one caption per line with no commentary."
                },
                {
                    "role": "user",
                    "content": f"Generate {num_prompts} diverse image captions that feature: {target_concepts_str}. Each caption should prominently include one or more of these concepts in different contexts, scenes, and artistic styles."
                }
            ]
            
            outputs = openai.ChatCompletion.create(
                model="gpt-4", messages=messages
            ).choices[0].message.content.lower().split("\n")
        else:
            # Meta Llama implementation
            model = transformers.pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            messages = [
                {
                    "role": "user",
                    "content": f"Generate {num_prompts} diverse image captions that feature {target_concepts_str}. Each caption should prominently include these concepts in various contexts, scenes, and artistic styles. Output one caption per line."
                }
            ]
            terminators = [
                model.tokenizer.eos_token_id,
                model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model(
                messages,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"][-1]['content'].split("\n")[1:-1]
        
        class_prompt_collection = clean_prompt([x for x in outputs if x != ''])[:num_prompts]
        
        # Save prompts
        with open(concept["class_prompt"], "w") as f:
            for prompt in class_prompt_collection:
                f.write(prompt.strip() + "\n")
        print(f"Saved {len(class_prompt_collection)} unlearning prompts to '{concept['class_prompt']}'.")
    else:
        with open(concept["class_prompt"]) as f:
            class_prompt_collection = [x.strip() for x in f.readlines()]
    
    return class_prompt_collection


def _setup_generation_pipeline(args, accelerator):
    """Set up the diffusion pipeline for image generation."""
    torch_dtype = (
        torch.float16 if accelerator.device.type == "cuda" else torch.float32
    )
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16
        
    pipeline = DiffusionPipeline.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.revision,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)
    
    return pipeline


def _generate_images_from_prompts(args, num_new_images, concept, class_prompt_collection, 
                                class_images_dir, pipeline, accelerator, logger):
    """Generate unlearning images from the prompt collection."""
    logger.info(f"Number of unlearning images to sample: '{num_new_images}'.")

    sample_dataset = PromptDataset(class_prompt_collection, num_new_images)
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset, batch_size=args.sample_batch_size
    )
    sample_dataloader = accelerator.prepare(sample_dataloader)

    # Clean up existing files
    for filename in ["captions.txt", "images.txt"]:
        filepath = f"{class_images_dir}/{filename}"
        if os.path.exists(filepath):
            print(f"Removing existing file: '{filepath}'")
            os.remove(filepath)

    # For FMN unlearning, we don't use negative prompts - we want the target concepts
    negative_prompt = None
    
    # Generate images
    for example in tqdm(
        sample_dataloader,
        desc="Generating unlearning images",
        disable=not accelerator.is_local_main_process,
    ):
        accelerator.wait_for_everyone()
        with open(f"{class_images_dir}/captions.txt", "a") as f1, open(
            f"{class_images_dir}/images.txt", "a"
        ) as f2:
            images = pipeline(
                example["prompt"],
                num_inference_steps=100,
                guidance_scale=6.0,
                negative_prompt=[negative_prompt]*len(example["prompt"]) if negative_prompt else None,
                eta=1.0,
            ).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = (
                    class_images_dir
                    / f"images/{example['index'][i]+1}-{hash_image}.jpg"
                )
                image.save(image_filename)
                f2.write(str(image_filename) + "\n")
            f1.write("\n".join(example["prompt"]) + "\n")
            accelerator.wait_for_everyone()


def _update_concept_paths(concept, class_images_dir):
    """Update concept dictionary with generated image paths."""
    concept["class_prompt"] = os.path.join(class_images_dir, "captions.txt")
    concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
    concept["instance_prompt"] = os.path.join(class_images_dir, "captions.txt")
    concept["instance_data_dir"] = os.path.join(class_images_dir, "images.txt")


def clean_prompt(class_prompt_collection):
    """Clean and normalize prompts."""
    class_prompt_collection = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', "") for x in class_prompt_collection]
    return class_prompt_collection


def setup_data_and_scheduler(args, tokenizer, accelerator, optimizer):
    """
    Set up dataset, dataloader, and learning rate scheduler for FMN unlearning.
    """
    # Create dataset with ForgetMeNotDataset for unlearning
    num_class_images = min(args.train_size, args.num_class_images) if args.with_prior_preservation else 0
    print(f"\tCreating FMN unlearning dataset with '{len(args.concepts_list)}' concepts...")
    
    train_dataset = ForgetMeNotDataset(
        concepts_list=args.concepts_list,
        concept_type=args.concept_type,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
    )

    print(f"\tCreating training dataloader with '{len(train_dataset)}' examples and batch size '{args.train_batch_size}'...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Calculate training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    print(f"\tNumber of training steps: '{args.max_train_steps}'")

    # Create learning rate scheduler
    print(f"\tUsing lr scheduler: '{args.lr_scheduler}'")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Store for later use
    args._overrode_max_train_steps = overrode_max_train_steps
    args._train_dataset = train_dataset

    return train_dataloader, lr_scheduler