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
    input_anchor_ids = [example["instance_anchor_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    image_paths = [example["image_path"] for example in examples]
    target_prompts = [example["target_prompt"] for example in examples]
    anchor_prompts = [example["anchor_prompt"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    input_anchor_ids = torch.cat(input_anchor_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "input_anchor_ids": input_anchor_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1),
        "image_paths": image_paths,
        "target_prompts": target_prompts,
        "anchor_prompts": anchor_prompts,
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


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        concept_type,
        with_style_replacement,
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
        self.with_style_replacement = with_style_replacement

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            inst_img_path = [
                (x, y, concept["caption_target"])
                for (x, y) in zip(inst_images_path, inst_prompt)
            ]
            self.instance_images_path.extend(inst_img_path)

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
        if self.concept_type == "style":
            # Clean up instance_target
            instance_target = instance_target.replace("_", " ")
            instance_target = instance_target.replace(" Style", "")

            # Replace style in prompt or append it
            if self.with_style_replacement:
                # Replace the style in the prompt with the target style
                instance_prompt = re.sub(
                    r"in\s+.*?style", f"in {instance_target} style", instance_prompt, flags=re.IGNORECASE
                )
                if instance_prompt == instance_target:
                    instance_prompt = f"An image in {instance_target} Style"
                    print(f"Unsuccessful replacement for style '{instance_target}' in prompt '{instance_prompt}'. Using '{instance_prompt}' instead.")
            else:
                r = np.random.choice([0, 1])
                instance_prompt = instance_prompt.replace(".", "")
                instance_prompt = (
                    f"{instance_prompt}, in {instance_target} Style"
                    if r == 0
                    else f"In {instance_target} Style, {instance_prompt}"
                )
        elif self.concept_type in ["nudity", "inappropriate_content"]:
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_target}, {instance_prompt}"
                if r == 0
                else f"in {instance_target} style, {instance_prompt}"
                if r == 1
                else f"{instance_prompt}, {instance_target}"
            )
        elif self.concept_type == "object":
            anchor, target = instance_target.split("+")
            if anchor == "*":
                instance_prompt = target
            else:
                # Replace anchor with target in prompt
                instance_prompt = re.sub(re.escape(anchor), target, instance_prompt, flags=re.IGNORECASE)
                if instance_prompt == anchor:
                    instance_prompt = f"A {target} image"
                    print(f"Unsuccessful replacement for anchor '{anchor}' in prompt '{instance_prompt}'. Using '{instance_prompt}' instead.")
        elif self.concept_type == "memorization":
            instance_prompt = instance_target.split("+")[1]
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
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ";" in instance_target:
            instance_target = instance_target.split(";")
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        example["anchor_prompt"] = instance_anchor_prompt
        example["target_prompt"] = instance_prompt

        # apply resize augmentation and create a valid image region mask
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
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

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


def isimage(path):
    if "png" in path.lower() or "jpg" in path.lower() or "jpeg" in path.lower():
        return True


def filter(
    folder,
    impath,
    outpath=None,
    unfiltered_path=None,
    threshold=0.15,
    image_threshold=0.5,
    anchor_size=10,
    target_size=3,
    return_score=False,
):
    model = torch.jit.load(
        "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt"
    )
    if isinstance(folder, list):
        image_paths = folder
        image_captions = ["None" for _ in range(len(image_paths))]
    elif Path(folder / "images.txt").exists():
        with open(f"{folder}/images.txt", "r") as f:
            image_paths = f.read().splitlines()
        with open(f"{folder}/captions.txt", "r") as f:
            image_captions = f.read().splitlines()
    else:
        image_paths = [
            os.path.join(str(folder), file_path)
            for file_path in os.listdir(folder)
            if isimage(file_path)
        ]
        image_captions = ["None" for _ in range(len(image_paths))]

    batch = small_288(Image.open(impath).convert("RGB")).unsqueeze(0)
    embedding_target = model(batch)[0, :]

    filtered_paths = []
    filtered_captions = []
    unfiltered_paths = []
    unfiltered_captions = []
    count_dict = {}
    for im, c in zip(image_paths, image_captions):
        if c not in count_dict:
            count_dict[c] = 0
        if isinstance(folder, list):
            batch = small_288(im).unsqueeze(0)
        else:
            batch = small_288(Image.open(im).convert("RGB")).unsqueeze(0)
        embedding = model(batch)[0, :]

        diff_sscd = (embedding * embedding_target).sum()

        if diff_sscd <= image_threshold:
            filtered_paths.append(im)
            filtered_captions.append(c)
            count_dict[c] += 1
        else:
            unfiltered_paths.append(im)
            unfiltered_captions.append(c)

    # only return score
    if return_score:
        score = len(unfiltered_paths) / (len(unfiltered_paths) + len(filtered_paths))
        return score

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/samples", exist_ok=True)
    with open(f"{outpath}/captions.txt", "w") as f:
        for each in filtered_captions:
            f.write(each.strip() + "\n")

    with open(f"{outpath}/images.txt", "w") as f:
        for each in filtered_paths:
            f.write(each.strip() + "\n")
            imbase = Path(each).name
            shutil.copy(each, f"{outpath}/samples/{imbase}")

    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+ Filter Summary +")
    print(f"+ Remained images: {len(filtered_paths)}")
    print(f"+ Filtered images: {len(unfiltered_paths)}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")

    sorted_list = sorted(list(count_dict.items()), key=lambda x: x[1], reverse=True)
    anchor_prompts = [c[0] for c in sorted_list[:anchor_size]]
    target_prompts = [c[0] for c in sorted_list[-target_size:]]
    return anchor_prompts, target_prompts, len(filtered_paths)


def generate_anchor_images_if_needed(args, accelerator, logger):
    """
    Generate anchor images if prior preservation is enabled and images don't exist.

    Args:
        args: Arguments object containing configuration parameters
        accelerator: Accelerator object for distributed training
        logger: Logger for info messages
        
    Returns:
        None: Modifies args.concepts_list in place
    """
    # Generate class images if prior preservation is enabled.
    print(f"Processing '{len(args.concepts_list)}' concepts.")

    # Build string of concepts to exclude from prompt generation
    excluded_concepts = []
    for i, concept in enumerate(args.concepts_list):
        excluded_concepts.append(
            concept["caption_target"].replace("_", " ").replace(" Style", "")
        )
    if args.previously_unlearned:
        excluded_concepts.extend(
            ast.literal_eval(args.previously_unlearned)
        )
    if len(excluded_concepts) > 1:
        excluded_concepts_str = ", ".join(excluded_concepts[:-1]) + ", or " + excluded_concepts[-1]
    else:
        excluded_concepts_str = ", ".join(excluded_concepts)

    for i, concept in enumerate(args.concepts_list):
        # directly path to ablation images and its corresponding prompts is provided.
        if (
            concept["instance_prompt"] is not None
            and concept["instance_data_dir"] is not None
        ):
            break
        class_images_dir = Path(concept["class_data_dir"])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(f"{class_images_dir}/images", exist_ok=True)
        
        # we need to generate training images
        image_dir = (Path(os.path.join(class_images_dir, "images")))
        num_existing_class_images = len(list(image_dir.iterdir()))
        print(f"\t{i+1}. '{concept['caption_target']}': '{num_existing_class_images}' class images at '{image_dir}'.")
        if (
            num_existing_class_images < args.num_class_images
        ):  
            num_to_generate = args.num_class_images - num_existing_class_images
            print(f"Only '{num_existing_class_images}' class images found at '{image_dir}'. Generating '{num_to_generate}' more images...")
            pipeline = _setup_generation_pipeline(args, accelerator)
            class_prompt_collection = _get_class_prompt_collection(
                args, concept, excluded_concepts_str, class_images_dir, pipeline, accelerator
            )
            _generate_images_from_prompts(
                args, num_to_generate, concept, class_prompt_collection, class_images_dir, 
                pipeline, accelerator, logger
            )
            del pipeline

        # Handle memorization filtering
        if args.concept_type == "memorization":
            class_images_dir = _handle_memorization_filtering(
                args, concept, class_images_dir
            )

        # Update concept paths
        _update_concept_paths(concept, class_images_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _get_class_prompt_collection(args, concept, excluded_concepts_str, class_images_dir, pipeline, accelerator):
    """Get collection of prompts for class image generation."""
    # need to create prompts using class_prompt.
    if not os.path.isfile(concept["class_prompt"]):
        # LLM based prompt collection for ablating new objects or memorization images
        class_prompt = concept['caption_target']
        print(f"Generating anchor prompts for '{class_prompt}' to: '{concept['class_prompt']}'")
        # in case of object query chatGPT to generate captions containing the anchor category
        class_prompt_collection, caption_target = getanchorprompts(
            pipeline,
            accelerator,
            class_prompt,
            excluded_concepts_str,
            args.concept_type,
            class_images_dir,
            args.num_class_prompts,
            mem_impath=args.mem_impath if args.concept_type == "memorization" else None,
            model_id=args.prompt_gen_model
        )
        with open(concept["class_prompt"], "w") as f:
            for prompt in class_prompt_collection:
                f.write(prompt.strip() + "\n")
        print(f"Saved {len(class_prompt_collection)} anchor prompts to '{concept['class_prompt']}'.")
    # class_prompt is filepath to prompts.
    else:
        with open(concept["class_prompt"]) as f:
            class_prompt_collection = [x.strip() for x in f.readlines()]
    
    return class_prompt_collection


def _generate_images_from_prompts(args, num_new_images, concept, class_prompt_collection, 
                                class_images_dir, pipeline, accelerator, logger):
    """Generate images from the prompt collection."""
    logger.info(f"Number of class images to sample: '{num_new_images}'.")

    sample_dataset = PromptDataset(class_prompt_collection, num_new_images)
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset, batch_size=args.sample_batch_size
    )
    sample_dataloader = accelerator.prepare(sample_dataloader)

    # Clean up existing files
    if os.path.exists(f"{class_images_dir}/captions.txt"):
        print(f"Removing existing captions file: '{class_images_dir}/captions.txt'")
        os.remove(f"{class_images_dir}/captions.txt")
    if os.path.exists(f"{class_images_dir}/images.txt"):
        print(f"Removing existing images file: '{class_images_dir}/images.txt'")
        os.remove(f"{class_images_dir}/images.txt")

    negative_prompt = None
    # Build negative prompt for anchor dataset (prevent UA rebounding)
    if args.with_negative_prompt:
        # Append previously unlearned concepts
        if args.previously_unlearned:
            negative_prompt_list = ast.literal_eval(args.previously_unlearned)
        else: 
            negative_prompt_list = []
        
        if args.concept_type == "style":
            # Add styles to unlearn this training run
            for concept in args.concepts_list:
                negative_prompt_list.append(concept["caption_target"])

            # Add " Style" suffix to each negative prompt
            for i in range(len(negative_prompt_list)):
                negative_prompt_list[i] = f"{negative_prompt_list[i]} Style"
        elif args.concept_type == "object":
            # Add objects to unlearn this training run
            for concept in args.concepts_list:
                target_object = concept["caption_target"].split("+")[1]
                negative_prompt_list.append(target_object)
        else:
            raise ValueError(f"'{args.concept_type}' not supported for negative prompting.")
        
        # Convert list to string
        negative_prompt = ", ".join(negative_prompt_list)
        print(f"Using negative prompt: '{negative_prompt}'")
    
    for example in tqdm(
        sample_dataloader,
        desc="Generating class images",
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


def _handle_memorization_filtering(args, concept, class_images_dir):
    """Handle memorization filtering if needed."""
    filter(
        class_images_dir,
        args.mem_impath,
        outpath=str(class_images_dir / "filtered"),
    )
    with open(class_images_dir / "caption_target.txt", "r") as f:
        concept["caption_target"] = f.readlines()[0].strip()
    return class_images_dir / "filtered"


def _update_concept_paths(concept, class_images_dir):
    """Update concept dictionary with generated image paths."""
    concept["class_prompt"] = os.path.join(class_images_dir, "captions.txt")
    concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
    concept["instance_prompt"] = os.path.join(class_images_dir, "captions.txt")
    concept["instance_data_dir"] = os.path.join(class_images_dir, "images.txt")
    
def getanchorprompts(
    pipeline,
    accelerator,
    class_prompt,
    excluded_concepts_str,
    concept_type,
    class_images_dir,
    num_class_images=200,
    mem_impath=None,
    model_id="meta-llama",
):
    if model_id == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    class_prompt_collection = []
    caption_target = []
    if concept_type in ["object", "nudity", "inappropriate_content", "style"]:
        if model_id == "openai":
            if concept_type == "style":
                messages = [
                    {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line."},
                    {"role": "user", "content": f"Generate {num_class_images} random prompts in the format 'A {{object}} image in {{style}} style'. Don't include any prompts with styles similar to {excluded_concepts_str}."},
                ]
            elif concept_type == "object":
                class_prompt = class_prompt.split("+")[0]
                messages = [
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
                            f"Generate {num_class_images} short captions for images that feature exactly one {class_prompt}. "
                            f"The image must not contain any other animals, humans, or objects. "
                            f"Each caption must include the word \"{class_prompt}\". "
                            f"The caption should be general enough that replacing the word \"{class_prompt}\" with another animal, human, or object would still result in a plausible description."
                        )
                    }
                ]
            print(f"Using GPT query: '{messages[1]['content']}'")
        else:
            messages = [
                    {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
                    {"role": "user", "content": f'''Generate {num_class_images} caption for images containing a {class_prompt}. The caption must also contain the word "{class_prompt}". DO NOT add any unnecessary adjectives or emotion words in the caption. Please keep the caption factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation.

                            Example captions for the category "cat" are:
                            1. A photo of a siamese cat playing in a garden.
                            2. A cat is sitting beside a book in a library.
                            4. Watercolor style painting of a cat. '''
                    }, ]
        numtries = 1
        while True:
            if model_id == "openai":
                outputs = openai.ChatCompletion.create(
                    model="gpt-4.1", messages=messages
                ).choices[0].message.content.lower().split("\n")
            else:
                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = model(
                    messages,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )[0]["generated_text"][-1]['content'].split("\n")[1:-1]

            print(f"{numtries}: {len(outputs)} outputs generated.")
            if concept_type in ["object", "nudity", "inappropriate_content", "style"]:
                class_prompt_collection += [
                    x for x in outputs if x != ''
                ]
            else:
                class_prompt_collection += [
                    x
                    for x in outputs
                    if (class_prompt in x and x != '')
                ]
            messages.append(
                {"role": "assistant", "content": "\n".join(outputs)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Generate {num_class_images-len(class_prompt_collection)} more captions",
                }
            )
            messages = messages[min(len(messages),-10):]
            numtries +=1
            if len(class_prompt_collection) >= num_class_images or numtries > 10:
                break
        class_prompt_collection = clean_prompt(class_prompt_collection)[
            :num_class_images
        ]  
    elif concept_type == "memorization":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        num_prompts_firstpass = 5
        num_prompts_secondpass = 2
        threshold = 0.3
        # Generate num_prompts_firstpass paraphrases which generate different content at least 1-threshold % of the times.
        os.makedirs(class_images_dir / "temp/", exist_ok=True)
        class_prompt_collection_counter = []
        caption_target = []
        prev_captions = []
        messages = [
            {
                "role": "user",
                "content": f"Generate {4*num_prompts_firstpass} different paraphrase of the caption: {class_prompt}. Preserve the meaning when paraphrasing.",
            }
        ]
        while True:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            # print(completion.choices[0].message.content.lower().split('\n'))
            class_prompt_collection_ = [
                x.strip()
                for x in completion.choices[0].message.content.lower().split("\n")
                if x.strip() != ""
            ]
            class_prompt_collection_ = clean_prompt(class_prompt_collection_)
            # print(class_prompt_collection_)
            for prompt in tqdm(
                class_prompt_collection_,
                desc="Generating anchor and target prompts ",
                disable=not accelerator.is_local_main_process,
            ):
                print(f"Prompt: {prompt}")
                images = pipeline(
                    [prompt] * 10,
                    num_inference_steps=100,
                ).images

                score = filter(images, mem_impath, return_score=True)
                print(f"Memorization rate: {score}")
                if (
                    score <= threshold
                    and prompt not in class_prompt_collection
                    and len(class_prompt_collection) < num_prompts_firstpass
                ):
                    class_prompt_collection += [prompt]
                    class_prompt_collection_counter += [score]
                elif (
                    score >= 0.6
                    and prompt not in caption_target
                    and len(caption_target) < 2
                ):
                    caption_target += [prompt]
                if (
                    len(class_prompt_collection) >= num_prompts_firstpass
                    and len(caption_target) >= 2
                ):
                    break

            if len(class_prompt_collection) >= num_prompts_firstpass:
                break
            prev_captions += class_prompt_collection_
            prev_captions_ = ",".join(prev_captions[-40:])

            messages = [
                {
                    "role": "user",
                    "content": f"Generate {4*(num_prompts_firstpass- len(class_prompt_collection))} different paraphrase of the caption: {class_prompt}. Preserve the meaning the most when paraphrasing. Also make sure that the new captions are different from the following captions: {prev_captions_[:4000]}",
                }
            ]

        # Generate more paraphrases using the captions we retrieved above.
        for prompt in class_prompt_collection[:num_prompts_firstpass]:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate {num_prompts_secondpass} different paraphrases of: {prompt}. ",
                    }
                ],
            )
            class_prompt_collection += clean_prompt(
                [
                    x.strip()
                    for x in completion.choices[0].message.content.lower().split("\n")
                    if x.strip() != ""
                ]
            )

        for prompt in tqdm(
            class_prompt_collection[num_prompts_firstpass:],
            desc="Memorization rate for final prompts",
        ):
            images = pipeline(
                [prompt] * 10,
                num_inference_steps=100,
            ).images

            class_prompt_collection_counter += [
                filter(images, mem_impath, return_score=True)
            ]

        # select least ten and most memorized text prompts to be selected as anchor and target prompts.
        class_prompt_collection = sorted(
            zip(class_prompt_collection, class_prompt_collection_counter),
            key=lambda x: x[1],
        )
        caption_target += [x for (x, y) in class_prompt_collection if y >= 0.6]
        class_prompt_collection = [
            x for (x, y) in class_prompt_collection if y <= threshold
        ][:10]
        print("Anchor prompts:", class_prompt_collection)
        print("Target prompts:", caption_target)
    return class_prompt_collection, ";*+".join(caption_target)


def clean_prompt(class_prompt_collection):
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


def safe_dir(dir):
    if not dir.exists():
        os.makedirs(str(dir), exist_ok=True)
    return dir

def setup_data_and_scheduler(args, tokenizer, accelerator, optimizer):
    """
    Set up dataset, dataloader, and learning rate scheduler.
    
    Args:
        args: Arguments object
        tokenizer: Tokenizer
        accelerator: Accelerator object
        optimizer: Optimizer
        
    Returns:
        tuple: (train_dataloader, lr_scheduler) and modifies args with calculated steps
    """
    # Create dataset and dataloader
    num_class_images = min(args.train_size, args.num_class_images)
    print(f"Creating training dataset with '{num_class_images}' images...")
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        concept_type=args.concept_type,
        with_style_replacement=args.with_style_replacement,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
    )

    print(f"Creating training dataloader with '{len(train_dataset)}' examples and batchsize '{args.train_batch_size}'...")
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
    print(f"Number of training steps: '{args.max_train_steps}'")

    # Create learning rate scheduler
    print(f"Using lr scheduler: '{args.lr_scheduler}'")
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