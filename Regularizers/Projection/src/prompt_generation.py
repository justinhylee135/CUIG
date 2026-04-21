import ast
import gc
import os
import random
import re

import numpy as np
import openai
import torch
import transformers


def generate_auxiliary_prompts(auxiliary_prompts_path, num_prompts, concept_type, previously_unlearned, target_concepts, dual_domain=True, llm_model_id="openai"):
    """
    Generate auxiliary prompts using gpt and write to auxiliary_prompts_path.
    Args:
        auxiliary_prompts_path (str): Path to save the generated auxiliary prompts.
        num_prompts (int): Number of prompts to generate.
        concept_type (str): Type of concept to generate prompts for.
        previously_unlearned (list): List of previously unlearned concepts.
        target_concepts (list): Current concepts to unlearn
        dual_domain (bool): Whether to generate prompts for both style and object domain.
    """
    local_llm_model = None
    local_llm_tokenizer = None
    use_openai = llm_model_id == "openai"
    resolved_llm_model_id = llm_model_id

    def _cleanup_local_llm():
        nonlocal local_llm_model, local_llm_tokenizer
        if local_llm_model is not None:
            del local_llm_model
            local_llm_model = None
        if local_llm_tokenizer is not None:
            del local_llm_tokenizer
            local_llm_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _postprocess_llm_lines(raw_lines):
        cleaned_lines = []
        for line in raw_lines:
            line = line.strip()
            line = re.sub(r"^\s*[-*]\s*", "", line)
            line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)
            if line:
                cleaned_lines.append(line)
        return cleaned_lines

    def _normalize_prompt_text(prompt):
        prompt = re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        prompt = re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        prompt = prompt.strip()
        prompt = prompt.replace('"', "")
        return prompt

    def _generate_prompts_with_llm(llm_query, num_prompts_needed, excluded_concepts):
        generated_prompts_list = []
        generated_prompts_set = set()
        num_llm_calls = 1

        while len(generated_prompts_list) < num_prompts_needed and num_llm_calls <= 10:
            print(f"{num_llm_calls}. Querying '{resolved_llm_model_id}' for '{num_prompts_needed - len(generated_prompts_list)}' prompts...")

            try:
                if use_openai:
                    llm_response = openai.ChatCompletion.create(
                        model=resolved_llm_model_id,
                        messages=llm_query,
                        temperature=0,
                    )
                    llm_outputs = llm_response.choices[0].message.content.split("\n")
                    llm_query.append({"role": "assistant", "content": llm_response.choices[0].message.content})
                else:
                    prompt_text = local_llm_tokenizer.apply_chat_template(
                        llm_query,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    model_inputs = local_llm_tokenizer(
                        prompt_text,
                        return_tensors="pt",
                    )
                    if torch.cuda.is_available():
                        model_inputs = {k: v.to(local_llm_model.device) for k, v in model_inputs.items()}

                    terminators = [local_llm_tokenizer.eos_token_id]
                    eot_id = local_llm_tokenizer.convert_tokens_to_ids("<|im_end|>")
                    if eot_id is not None:
                        terminators.append(eot_id)

                    with torch.no_grad():
                        generated_ids = local_llm_model.generate(
                            **model_inputs,
                            max_new_tokens=2048,
                            do_sample=False,
                            num_beams=1,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            eos_token_id=terminators,
                            pad_token_id=local_llm_tokenizer.eos_token_id,
                        )

                    new_token_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
                    decoded_output = local_llm_tokenizer.decode(new_token_ids, skip_special_tokens=True)
                    llm_outputs = _postprocess_llm_lines(decoded_output.split("\n"))
                newly_generated_prompts = []
                batch_seen_prompts = set()
                for prompt in llm_outputs:
                    normalized_prompt = _normalize_prompt_text(prompt)
                    if normalized_prompt == "":
                        continue
                    if not all(exclusion.lower() not in normalized_prompt.lower() for exclusion in excluded_concepts):
                        continue
                    if normalized_prompt in generated_prompts_set or normalized_prompt in batch_seen_prompts:
                        continue
                    batch_seen_prompts.add(normalized_prompt)
                    newly_generated_prompts.append(normalized_prompt)

                if use_openai:
                    llm_query.append({"role": "assistant", "content": "\n".join(newly_generated_prompts)})
                else:
                    llm_query.append({"role": "assistant", "content": "\n".join(newly_generated_prompts)})

                generated_prompts_list.extend(newly_generated_prompts)
                generated_prompts_set.update(newly_generated_prompts)

                llm_query.append({
                    "role": "user",
                    "content": f"Generate {num_prompts_needed - len(generated_prompts_list)} more captions in the same format",
                })
                llm_query = llm_query[-10:]

            except Exception as e:
                print(f"Error querying {resolved_llm_model_id}: {e}")
                break

            num_llm_calls += 1

        return generated_prompts_list[:num_prompts_needed]

    if use_openai:
        resolved_llm_model_id = "gpt-4.1"
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        if llm_model_id == "mistral":
            resolved_llm_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            transformers.AutoConfig.from_pretrained(resolved_llm_model_id, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Invalid HuggingFace model_id '{resolved_llm_model_id}': {e}") from e

        if torch.cuda.is_available():
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)
        local_llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            resolved_llm_model_id,
            trust_remote_code=True,
        )
        qwen_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        local_llm_model = transformers.AutoModelForCausalLM.from_pretrained(
            resolved_llm_model_id,
            torch_dtype=qwen_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        local_llm_model.eval()

    print(f"Using LLM Model: {resolved_llm_model_id}")

    try:
        # Create list of concepts to exclude from prompt generation
        if isinstance(previously_unlearned, str):
            previously_unlearned = ast.literal_eval(previously_unlearned) if "[" in previously_unlearned else [previously_unlearned]
        elif not previously_unlearned:
            previously_unlearned = []

        # Concepts we're unlearning this training run
        no_duplicates = list(dict.fromkeys(target_concepts))
        if len(no_duplicates) < len(target_concepts):
            print(
                f"Removed '{len(target_concepts) - len(no_duplicates)}' duplicate concepts from '{target_concepts}' to '{no_duplicates}'"
            )
            target_concepts = no_duplicates

        # Remove unnecessary prefix and suffix from target concepts
        for i in range(len(target_concepts)):
            target_concepts[i] = target_concepts[i].replace(" Style", "")
            target_concepts[i] = target_concepts[i].replace("An image of ", "")
            print(f"Excluding for gradient projection prompts: '{target_concepts[i]}'")

        # Generate style prompts
        print("\n=== Generating Style Prompts ===")
        style_prompts = []

        # Add previously unlearned styles to prompts for projection
        if concept_type == "style":
            for concept in previously_unlearned:
                if concept == "":
                    continue
                prompt = concept.replace("_", " ")
                prompt = f"An image in the style of {prompt}"
                print(f"Adding previously unlearned concept to prompt: '{prompt}'")
                style_prompts.append(prompt)
            print(f"Added '{len(style_prompts)}' previously unlearned style prompts")

        # Ensure we don't project the concept we're going to unlearn (then we can't unlearn it)
        style_exclusions = target_concepts if concept_type == "style" else []

        # Build LLM Query
        style_messages = [
            {
                "role": "system",
                "content": "You are an expert at generating creative image captions. You will generate random prompts in the format 'An image in the style of {style}'. Prefer popular, well-known art styles, major artistic movements, and famous artists with distinctive visual language. Remove unnecessary suffixes like 'style' or 'art' from style names. Avoid obscure niche references, media formats, photography modes, and design formats. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line. Ensure no duplicate prompts.",
            },
            {
                "role": "user",
                "content": (
                    f'Generate {num_prompts} random prompts for images in the format "An image in the style of {{style}}". The styles should be diverse art styles, artist names, or artistic movements. None of the styles should contain the words: {style_exclusions}.'
                    if style_exclusions
                    else f'Generate {num_prompts} random prompts for images in the format "An image in the style of {{style}}". The styles should be diverse art styles, artist names, or artistic movements.'
                ),
            },
        ]

        # Generate style prompt if our concept type is style or if we're doing both domains (style/object)
        if concept_type == "style" or dual_domain:
            print(f"Style LLM Prompt: '{style_messages[1]['content']}'")

            # If we're doing both domnains, only half the total prompts will be style, other will be object
            if dual_domain:
                num_style_prompts = (num_prompts // 2) - len(style_prompts)
            else:
                num_style_prompts = num_prompts - len(style_prompts)
            style_prompts.extend(_generate_prompts_with_llm(style_messages, num_style_prompts, style_exclusions))
        else:
            print(f"Skipping style prompt generation for concept type '{concept_type}'")

        # Generate object prompts
        print("\n=== Generating Object Prompts ===")
        object_prompts = []

        # Add previously unlearned objects to prompt collection
        if concept_type != "style":
            for concept in previously_unlearned:
                if concept == "":
                    continue
                prompt = f"An image of a {concept}"
                print(f"Adding previously unlearned object prompt: '{prompt}'")
                object_prompts.append(prompt)
            print(f"Added '{len(object_prompts)}' previously unlearned object prompts")

        object_exclusions = target_concepts if concept_type != "style" else []
        object_messages = [
            {
                "role": "system",
                "content": "You are an expert at generating creative image captions. You will generate random prompts in the format 'An image of a {object}'. Prefer simple, common, visually recognizable everyday objects, animals, vehicles, buildings, tools, foods, and natural things. Avoid scenes, relations, rare niche items, and overly specific compositions. You generate strictly formatted image captions without any commentary, explanations, or introductions. You just output the prompts, one per line. Ensure no duplicate prompts.",
            },
            {
                "role": "user",
                "content": (
                    f'Generate {num_prompts} random prompts for images in the format "An image of a {{object}}". The objects should be diverse, concrete things like animals, vehicles, buildings, tools, etc. None of the objects should contain the words: {object_exclusions}.'
                    if object_exclusions
                    else f'Generate {num_prompts} random prompts for images in the format "An image of a {{object}}". The objects should be diverse, concrete things like animals, vehicles, buildings, tools, etc.'
                ),
            },
        ]

        # Generate style prompt if our concept type is not style or if we're doing both domains (style/object)
        if concept_type != "style" or dual_domain:
            print(f"Object LLM Prompt: '{object_messages[1]['content']}'")
            if dual_domain:
                num_object_prompts = (num_prompts // 2) - len(object_prompts)
            else:
                num_object_prompts = num_prompts - len(object_prompts)
            object_prompts.extend(_generate_prompts_with_llm(object_messages, num_object_prompts, object_exclusions))
        else:
            print(f"Skipping object prompt generation for concept type '{concept_type}'")

        # Concatenate style and object prompts
        all_prompts = style_prompts + object_prompts

        # Clean up prompts
        all_prompts = list(dict.fromkeys(clean_prompt(all_prompts)))

        # Save prompts to prompts_path
        os.makedirs(os.path.dirname(auxiliary_prompts_path), exist_ok=True)
        with open(auxiliary_prompts_path, "w") as f:
            for prompt in all_prompts:
                f.write(prompt + "\n")

        # Status
        print(f"\nGenerated '{len(style_prompts)}' style prompts and '{len(object_prompts)}' object prompts")
        print(f"Total: '{len(all_prompts)}' prompts saved to '{auxiliary_prompts_path}'")

        return all_prompts
    finally:
        if not use_openai:
            _cleanup_local_llm()

def clean_prompt(class_prompt_collection):
    # Remove all digits from prompts
    class_prompt_collection = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in class_prompt_collection
    ]

    # Remove leading dots from prompts
    class_prompt_collection = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    
    # Strip leading/trailing whitespace from all prompts
    class_prompt_collection = [x.strip() for x in class_prompt_collection]

    # Remove double quotes from all prompts
    class_prompt_collection = [x.replace('"', "") for x in class_prompt_collection]

    return class_prompt_collection
