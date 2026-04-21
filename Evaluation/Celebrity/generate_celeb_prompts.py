"""Utility to request GPT-generated prompts for celebrity evaluation.

Usage:
    python generate_prompts.py --prompt "Morgan Freeman" --num_prompts 200 --output_path prompts/Morgan_Freeman.txt

Safeguards:
    * The script skips generation when the output file already exists.
    * GPT is instructed to emit only raw captions (one per line) without introductions.
    * Post-processing removes any stray introductions (e.g., "Sure! Here are...") before saving.
"""

import argparse
import os
import re

import openai
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a caption generator. Output ONLY standalone image captions, one per line, "
    "with no introductions, explanations, numbering, or additional commentary."
)

USER_PROMPT_TEMPLATE = (
    "Generate {count} unique captions for images containing {celeb}. "
    "Each caption must include the word \"{celeb}\" and be on its own line. "
    "Do not prepend any introductions or follow-up sentences."
)

DISALLOWED_PREFIXES = ("sure", "here are", "absolutely", "of course", "certainly")


def clean_prompts(prompts, celeb):
    """Remove numbering/quotes and trim whitespace from prompts."""
    cleaned = []
    for prompt in prompts:
        prompt = re.sub(r"[0-9]+", "", prompt)
        prompt = re.sub(r"^\.+", "", prompt)
        prompt = prompt.strip().replace('"', "")
        if not prompt:
            continue
        lowered = prompt.lower()
        if any(lowered.startswith(prefix) for prefix in DISALLOWED_PREFIXES):
            continue
        if celeb.lower() not in lowered:
            continue
        cleaned.append(prompt)
    return cleaned


def request_prompts(celeb, target_count):
    celeb_prompts = []
    celeb = celeb.replace("_", " ")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(count=target_count, celeb=celeb),
        },
    ]

    with tqdm(total=target_count, desc="Generating prompts") as pbar:
        while len(celeb_prompts) < target_count:
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            raw_output = completion.choices[0].message.content
            new_prompts = [
                prompt.strip()
                for prompt in raw_output.split("\n")
                if celeb.lower() in prompt.lower()
            ]
            celeb_prompts.extend(new_prompts)
            pbar.update(len(new_prompts))

            messages.append({"role": "assistant", "content": raw_output})
            remaining = target_count - len(celeb_prompts)
            if remaining <= 0:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Generate {remaining} more captions with no introductions. "
                        "Remember the captions must all be unique."
                    ),
                }
            )

    return clean_prompts(celeb_prompts, celeb)[:target_count]


def save_prompts(prompts, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(prompts))


def main():
    parser = argparse.ArgumentParser(description="Generate prompts via GPT for Celebrity evaluation.")
    parser.add_argument("--prompt", type=str, required=True, help="celeb subject for the prompts.")
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=50,
        help="Number of prompts to request from GPT.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path were the generated prompts will be stored.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        print(f"Prompt file already exists at {args.output_path}. Skipping generation.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable for celeb prompt generation.")

    openai.api_key = api_key
    prompts = request_prompts(args.prompt, args.num_prompts)
    save_prompts(prompts, args.output_path)
    print(f"Saved {len(prompts)} prompts for '{args.prompt}' to {args.output_path}.")


if __name__ == "__main__":
    main()
