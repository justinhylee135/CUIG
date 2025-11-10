import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from diffusers import DDIMScheduler
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers_utils import StableDiffusionWrapper, load_diffusers_model, sample_latents
from gen_embedding_matrix import (
    learn_k_means_from_input_embedding,
    save_embedding_matrix,
    search_closest_tokens,
)
from utils_exp import get_prompt, sanitize_filename, str2bool

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
    plt.close()


def get_models(pretrained_path: str, devices: List[torch.device], dtype: torch.dtype):
    model_orig = load_diffusers_model(pretrained_path, devices[1], dtype)
    model = load_diffusers_model(pretrained_path, devices[0], dtype)
    model_orig.eval()
    return model_orig, model

def save_to_dict(var, name, dict):
    if var is not None:
        if isinstance(var, torch.Tensor):
            var = var.cpu().detach().numpy()
        if isinstance(var, list):
            var = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in var]
    else:
        return dict
    
    if name not in dict:
        dict[name] = []
    
    dict[name].append(var)
    return dict


def parse_device_list(device_arg: str) -> List[torch.device]:
    tokens = [token.strip() for token in device_arg.split(",") if token.strip()]
    if not tokens:
        tokens = ["cuda:0" if torch.cuda.is_available() else "cpu"]

    devices: List[torch.device] = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower == "cpu":
            devices.append(torch.device("cpu"))
        elif token_lower.startswith("cuda"):
            devices.append(torch.device(token_lower))
        else:
            devices.append(torch.device(f"cuda:{int(token)}"))

    if len(devices) == 1:
        devices *= 2
    return devices[:2]


def resolve_dtype(requested: str, devices: Sequence[torch.device]) -> torch.dtype:
    request = (requested or "auto").lower()
    if request in ("auto", "default"):
        return torch.float16 if any(d.type == "cuda" for d in devices) and torch.cuda.is_available() else torch.float32

    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if request not in mapping:
        raise ValueError(f"Unsupported dtype '{requested}'. Choose from auto, float32, float16, bfloat16.")
    dtype = mapping[request]
    if dtype in (torch.float16, torch.bfloat16) and not torch.cuda.is_available():
        print(f"[AGE] WARNING: {request} requested without CUDA; falling back to float32.")
        return torch.float32
    return dtype



def train_age(
    prompt,
    train_method,
    start_guidance,
    negative_guidance,
    iterations,
    lr,
    pretrained_model,
    devices,
    seperator=None,
    image_size=512,
    ddim_steps=50,
    args=None,
):
    """
    Train AGE using a diffusers Stable Diffusion backbone.

    Parameters
    ----------
    prompt : str
        Concept to erase (e.g., "Van Gogh").
    train_method : str
        Parameter subset being optimized (noxattn, xattn, full, etc.).
    start_guidance : float
        Guidance scale for the positive sampling pass.
    negative_guidance : float
        Guidance scale for the erasure objective.
    iterations : int
        Number of optimization iterations.
    lr : float
        Learning rate for Adam.
    pretrained_model : str
        Diffusers repo id or path passed to `StableDiffusionPipeline`.
    devices : list[torch.device]
        Devices for the trainable and frozen pipelines.
    seperator : str, optional
        Optional delimiter for multi-concept prompts.
    image_size : int, optional
        Generated resolution (default 512).
    ddim_steps : int, optional
        Number of DDIM steps (default 50).
    """
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    # Acquire prompt for erasing and preserving
    prompt, preserved = get_prompt(prompt)

    # Separate for multi-concept unlearning
    if seperator is not None:
        erased_words = prompt.split(seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = preserved.split(seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    # Log
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    # Set dtype
    dtype = resolve_dtype(args.dtype, devices)
    print(f"[AGE] Using dtype {dtype} for UNet/VAE.")

    # Load model
    print("[AGE] Loading models...")
    model_orig, model = get_models(pretrained_model, devices, dtype)

    # Load scheduler
    print(f"[AGE] Setting up DDIM scheduler with '{ddim_steps}' steps...")
    scheduler = DDIMScheduler.from_config(model.scheduler_config)
    scheduler.set_timesteps(ddim_steps, device=devices[0])
    alphas_cumprod = scheduler.alphas_cumprod.to(devices[0], dtype=dtype)

    # choose parameters to train based on train_method
    parameters = []
    trainable_param_names: List[str] = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                parameters.append(param)
                trainable_param_names.append(name)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                parameters.append(param)
                trainable_param_names.append(name)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                parameters.append(param)
                trainable_param_names.append(name)
        if train_method == 'kv-xattn':
            if 'attn2' in name and ('to_k' in name or 'to_v' in name):
                parameters.append(param)
                trainable_param_names.append(name)
        # train only qkv layers in x attention layers
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                parameters.append(param)
                trainable_param_names.append(name)
                # return_nodes[name] = name
        # train all layers
        if train_method == 'full':
            # print(name)
            parameters.append(param)
            trainable_param_names.append(name)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                parameters.append(param)
                trainable_param_names.append(name)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:

                    parameters.append(param)
                    trainable_param_names.append(name)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:

                    parameters.append(param)
                    trainable_param_names.append(name)
    
    # Log parameters to update
    assert len(parameters) == len(trainable_param_names), "Mismatch between parameters and their names."
    print(f"Train Method: '{train_method}' -> '{len(parameters)}' parameters to optimize.")

    # Decodes latent into image and saves it
    def decode_and_save_image(model_orig, z, path):
        x = model_orig.decode_first_stage(z.to(model_orig.device))
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        img_np = (x[0].to(torch.float32).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(img_np)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()

    # Set model to train mode
    model.train()

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_latents(
        model,
        cond,
        image_size,
        image_size,
        ddim_steps,
        s,
        start_code=code,
        stop_step=t,
    )

    # Track metrics
    losses = []
    history_dict = {}

    # Set up optimizer
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    # Setup output path and
    name = f'diffusers-age-gumbel-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}-info_{args.info}'
    models_path = args.models_path
    os.makedirs(f'evaluation_folder/{name}', exist_ok=True)
    os.makedirs(f'invest_folder/{name}', exist_ok=True)
    os.makedirs(f'{models_path}/{name}', exist_ok=True)

    # Multiply by number of steps for pgd
    pbar = tqdm(range(args.pgd_num_steps*iterations))

    # prompt to embedding function
    def create_prompt(word):
        prompt = f'{word}'
        with torch.no_grad():
            emb = model.get_learned_conditioning([prompt]).detach()
        return emb

    # Random latent
    fixed_start_code = torch.randn((1, 4, 64, 64), device=devices[0], dtype=dtype)

    # create a matrix of embeddings for the entire vocabulary
    # Array is a list of the text embeddings while dict is the word to embedding mapping
    if not os.path.exists(f'{args.models_path}/embedding_matrix_dict_EN3K.pt'):
        print(f"Creating embedding matrix for EN3K vocabulary...")
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not os.path.exists(f'{args.models_path}/embedding_matrix_array_EN3K.pt'):
        print(f"Creating embedding matrix for EN3K vocabulary...")
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='EN3K')
    
    if not os.path.exists(f'{args.models_path}/embedding_matrix_array_Imagenet.pt'):
        print(f"Creating embedding matrix for Imagenet vocabulary...")
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='Imagenet')
    
    if not os.path.exists(f'{args.models_path}/embedding_matrix_dict_Imagenet.pt'):
        print(f"Creating embedding matrix for Imagenet vocabulary...")
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='dict', vocab='Imagenet')


    # Search the closest tokens in the vocabulary for each erased word, using the similarity matrix
    # if vocab in ['EN3K', 'Imagenet', 'CLIP'], then use the pre-defined vocabulary
    # if vocab in concept_dict.all_concepts, then use the custom concepts, i.e., 'nudity', 'artistic', 'human_body'
    # if vocab is 'keyword', then use the keywords in the erased words, defined in utils_concept.py

    # Load dictionary of vocab
    from utils_concept import ConceptDict
    concept_dict = ConceptDict()
    concept_dict.load_all_concepts()

    print('ignore_special_tokens:', args.ignore_special_tokens)
    
    all_sim_dict = dict()
    for word in erased_words:
        foundInConceptDict = False
        if args.vocab in ['EN3K', 'Imagenet', 'CLIP']:
            vocab = args.vocab
        elif args.vocab in concept_dict.all_concepts:
            # i.e., nudity, artistic, human_body
            print(f"Found '{args.vocab}' in concept dictionary.")
            vocab = concept_dict.get_concepts_as_dict(args.vocab)
            foundInConceptDict = True
        elif args.vocab == 'keyword':
            # i.e., 'Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn'
            vocab = concept_dict.get_concepts_as_dict(word)
            foundInConceptDict = True
        else:
            raise ValueError(f'Word {word} not found in concept dictionary, it should be either in EN3K, Imagenet, CLIP, or in the concept dictionary')
        
        # Extract top-k closest tokens and their similarity scores
        print(f"Using vocab: {args.vocab} for erased word: '{word}' and selecting top '{args.gumbel_k_closest}' closest tokens")
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, model, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=vocab, foundInConceptDict=foundInConceptDict)
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    # gumbel centers are learned from the closest tokens
    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    # create preserved set for each erased word
    for word in erased_words:
        print(f"Learning preserved set for erased word: '{word}' with '{args.gumbel_num_centers}' centers")
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

    # save preserved dict
    history_dict = save_to_dict(preserved_dict, f'preserved_set_0', history_dict)

    # create a matrix of embeddings for the preserved set
    print('Creating preserved matrix')
    weight_pi_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        preserved_matrix_list = []
        for i, word in enumerate(preserved_set):
            preserved_matrix_list.append(create_prompt(word))
            current_shape = torch.Size([i + 1, *preserved_matrix_list[-1].shape[1:]])
            print(i, word, current_shape)
        preserved_matrix = torch.cat(preserved_matrix_list, dim=0)
        preserved_matrix = preserved_matrix.flatten(start_dim=1).to(dtype=dtype, device=devices[0]).detach() # [n, 77*768]
        weight_pi = torch.full(
            (1, preserved_matrix.shape[0]),
            1 / preserved_matrix.shape[0],
            device=devices[0],
            dtype=torch.float32,
            requires_grad=True,
        )  # [1, n] kept in fp32 to stabilize Adam
        weight_pi_dict[erase_word] = weight_pi
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    print('weight_pi_dict:', weight_pi_dict)
    history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_0', history_dict)

    # optimizer for all pi vectors
    opt_weight_pi = torch.optim.Adam([weight_pi for weight_pi in weight_pi_dict.values()], lr=args.gumbel_lr)

    """
    Gumbel-Softmax function
        if `hard` is 1, then it is one-hot, if `hard` is 0, then it is a new soft version, which takes the top-k highest values and normalize them to 1
    """
    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10, k=args.gumbel_topk):
        logits_fp32 = logits.float()
        temp = torch.as_tensor(temperature, dtype=logits_fp32.dtype, device=logits_fp32.device)
        u = torch.rand_like(logits_fp32)
        # operate in fp32 to avoid log underflow/NaNs when logits are fp16
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits_fp32 + gumbel
        y = torch.nn.functional.softmax(y / temp, dim=-1)
        if hard == 1:
            y_hard = torch.zeros_like(logits_fp32)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        elif hard == 0:
            topk = min(k, y.shape[-1])
            top_k_values, _ = torch.topk(y, topk, dim=-1)
            top_k_mask = y >= top_k_values[..., -1].unsqueeze(-1)
            y = y * top_k_mask.float()
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        return y.to(dtype=logits.dtype)

    for i in pbar:
        # Sample word to erase
        word = random.sample(erased_words, 1)[0]

        # Determine whether to update pi or model parameters
        update_pi = (i % args.pgd_num_steps) != 0

        opt.zero_grad()
        model.zero_grad()
        model_orig.zero_grad()
        opt_weight_pi.zero_grad()

        # Concept to erase
        erase_concept = f"{word}"
        emb_erase = model.get_learned_conditioning([erase_concept])

        isDebug = args.debug_concept is not None
        debug_concept = args.debug_concept
        emb_debug = model.get_learned_conditioning([debug_concept])
        if isDebug: print(f"Replacing adaptive anchor with debug concept: '{debug_concept}'")

        # Concept to map to
        gumbel_weights = gumbel_softmax(weight_pi_dict[word])
        preserved_set = preserved_dict[word]
        mapped_idx = torch.argmax(gumbel_weights, dim=1).item()
        mapped_word = preserved_set[mapped_idx]
        pbar.set_description(f"erase:{word}→{mapped_word}")
        if not update_pi:
            gumbel_weights = gumbel_weights.detach()

        if isDebug:
            emb_anchor = emb_debug
            emb_anchor_sg = emb_debug.clone().detach()
        else:
            emb_anchor = torch.reshape(
                torch.matmul(
                    gumbel_weights.to(dtype=dtype),
                    preserved_matrix_dict[word].to(dtype=dtype),
                ).unsqueeze(0),
                (1, 77, 768),
            )
            emb_anchor_sg = emb_anchor.clone().detach()

        # Random timestep
        timestep_int = torch.randint(ddim_steps, (1,), device=devices[0])
        ddpm_timestep_lowerbound = round((int(timestep_int) / ddim_steps) * 1000)
        ddpm_timestep_upperbound = round((int(timestep_int + 1) / ddim_steps) * 1000)
        ddpm_timestep = torch.randint(ddpm_timestep_lowerbound, ddpm_timestep_upperbound, (1,), device=devices[0])

        # Random latent
        rnd_latent = torch.randn((1, 4, 64, 64), device=devices[0], dtype=dtype)

        with torch.no_grad():
            # Sample concept to erase
            erase_latent = quick_sample_till_t(emb_erase.to(devices[0]), start_guidance, rnd_latent, int(timestep_int))

            # Sample concept to anchor
            anchor_latent = quick_sample_till_t(emb_anchor.to(devices[0]), start_guidance, rnd_latent, int(timestep_int))

            # Get noise predictions using base model
            pnoise_erase_anchor_base = model_orig.apply_model(erase_latent.to(devices[1]), ddpm_timestep.to(devices[1]), emb_anchor_sg.to(devices[1]))
            pnoise_erase_erase_base = model_orig.apply_model(erase_latent.to(devices[1]), ddpm_timestep.to(devices[1]), emb_erase.to(devices[1]))
            pnoise_anchor_anchor_base = model_orig.apply_model(anchor_latent.to(devices[1]), ddpm_timestep.to(devices[1]), emb_anchor.to(devices[1]))

        # Double check to freeze gradients from base model pred
        pnoise_erase_anchor_base.requires_grad = False
        pnoise_erase_erase_base.requires_grad = False
        pnoise_anchor_anchor_base.requires_grad = False

        # Get noise predictions from fine-tuning model
        pnoise_erase_erase_ft = model.apply_model(erase_latent.to(devices[0]), ddpm_timestep.to(devices[0]), emb_erase.to(devices[0]))
        pnoise_anchor_anchor_ft = model.apply_model(anchor_latent.to(devices[0]), ddpm_timestep.to(devices[0]), emb_anchor.to(devices[0]))

        # Set ddim variables
        timestep_index = int(timestep_int)
        actual_step = int(scheduler.timesteps[timestep_index].item())
        alpha_bar_t = alphas_cumprod[actual_step].to(dtype=torch.float32)
        sqrt_alpha = torch.sqrt(alpha_bar_t).clamp(min=1e-6)
        sqrt_one_minus = torch.sqrt((1 - alpha_bar_t).clamp(min=1e-6))

        # Denoised latents from base model
        platent_erase_erase_base = (erase_latent.float() - sqrt_one_minus * pnoise_erase_erase_base.float()) / sqrt_alpha
        platent_erase_anchor_base = (erase_latent.float() - sqrt_one_minus * pnoise_erase_anchor_base.float()) / sqrt_alpha
        platent_anchor_anchor_base = (anchor_latent.float() - sqrt_one_minus * pnoise_anchor_anchor_base.float()) / sqrt_alpha

        # Denoised latents from fine-tuning model
        platent_erase_erase_ft = (erase_latent.float() - sqrt_one_minus * pnoise_erase_erase_ft.float()) / sqrt_alpha
        platent_anchor_anchor_ft = (anchor_latent.float() - sqrt_one_minus * pnoise_anchor_anchor_ft.float()) / sqrt_alpha

        if not update_pi: # Update model parameters
            loss = 0
            # AGE Loss
            loss += criteria(
                platent_erase_erase_ft.to(devices[0]),
                platent_erase_anchor_base.to(devices[0]) - (negative_guidance * (platent_erase_erase_base.to(devices[0]) - platent_erase_anchor_base.to(devices[0]))),
            )
            
            # Pnoise only
            # loss += criteria(
            #     pnoise_erase_erase_ft.to(devices[0]), pnoise_erase_anchor_base.to(devices[0])
            # )

            # Anchor preservation loss
            loss += args.lamda * criteria(platent_anchor_anchor_ft.to(devices[0]), platent_anchor_anchor_base.to(devices[0]))

            # Anchor preservation loss (pnoise only)
            # loss += args.lamda * criteria(pnoise_anchor_anchor_ft.to(devices[0]), pnoise_anchor_anchor_base.to(devices[0]))
            
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"model_loss": loss.item(), "timestep": int(timestep_int)})
            history_dict = save_to_dict(loss.item(), "loss", history_dict)
            opt.step()
        else: # Update pi vectors
            opt.zero_grad()
            opt_weight_pi.zero_grad()
            model.zero_grad()
            model_orig.zero_grad()
            loss = 0 
            loss -= criteria(platent_erase_erase_ft.to(devices[0]), platent_erase_anchor_base.to(devices[0]))
            loss -= args.lamda * criteria(platent_anchor_anchor_ft.to(devices[0]), platent_anchor_anchor_base.to(devices[0]))
            if torch.isnan(loss):
                print(f"[pi-update] loss is NaN at iter {i}; platent_erase_erase_ft range=({platent_erase_erase_ft.min().item():.4f}, {platent_erase_erase_ft.max().item():.4f}), "
                      f"platent_erase_anchor_base range=({platent_erase_anchor_base.min().item():.4f}, {platent_erase_anchor_base.max().item():.4f})")
            loss.backward()
            pbar.set_postfix({"pi_loss": loss.item(), "timestep": int(timestep_int)})
            opt_weight_pi.step()
            history_dict = save_to_dict([weight_pi_dict[word].cpu().detach().numpy(), i, preserved_set[torch.argmax(weight_pi_dict[word], dim=1)], word], 'weight_pi', history_dict)
            history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_{i}', history_dict)

        if args.save_freq is not None and i % (args.save_freq) == 0:
            model.eval()
            with torch.no_grad():
                for target in erased_words:
                    preserved_set = preserved_dict[target]
                    pi_anchor = preserved_set[torch.argmax(weight_pi_dict[target], dim=1)]
                    emb_pi_anchor = torch.reshape(
                        torch.matmul(
                            gumbel_softmax(weight_pi_dict[target]).to(dtype=dtype),
                            preserved_matrix_dict[target].to(dtype=dtype),
                        ).unsqueeze(0),
                        (1, 77, 768),
                    )

                    if isDebug:
                        pi_anchor = debug_concept
                        emb_pi_anchor = model.get_learned_conditioning([debug_concept])

                    emb_target = model.get_learned_conditioning([target])
                    sample_steps = 100
                    print(f"Saving evaluation images for Target: '{target}' mapped to Anchor: '{pi_anchor}' at iter '{i}' with '{sample_steps}' ddim steps and guidance '{start_guidance}'")

                    # fixed_start_code = torch.randn((1, 4, 64, 64), device=devices[0], dtype=dtype)
                    platent_rnd_pi_anchor = quick_sample_till_t(emb_pi_anchor.to(devices[0]), start_guidance, fixed_start_code, int(sample_steps))
                    decode_and_save_image(model_orig, platent_rnd_pi_anchor, path=sanitize_filename(f'evaluation_folder/{name}/epoch{i}_Anchor{pi_anchor}.png'))

                    platent_rnd_target = quick_sample_till_t(emb_target.to(devices[0]), start_guidance, fixed_start_code, int(sample_steps))
                    decode_and_save_image(model_orig, platent_rnd_target, path=sanitize_filename(f'evaluation_folder/{name}/epoch{i}_Target{target}.png'))
            model.train()

        # if i % 100 == 0:
        #     save_history(losses, name, word_print, models_path=models_path)
        #     torch.save(history_dict, f'invest_folder/{name}/history_dict_{i}.pt')

    model.eval()

    # save_model(model, name, models_path=models_path)
    save_trainable_unet(model, trainable_param_names, args.save_path, name)
    save_history(losses, name, word_print, models_path=models_path)


def save_model(model: StableDiffusionWrapper, name: str, models_path: str):
    target_dir = Path(models_path) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[AGE] Saving diffusers weights to {target_dir}")
    model.pipe.save_pretrained(target_dir)


def save_trainable_unet(model: StableDiffusionWrapper, param_names: List[str], save_path: Optional[str], run_name: str):
    if not param_names:
        print("[AGE] No trainable UNet parameters were selected; skipping filtered save.")
        return
    
    # Set path and dir
    target_file = Path(save_path)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract state dict
    state_dict = model.model.diffusion_model.state_dict()
    filtered_state = {k: v.cpu() for k, v in state_dict.items() if k in param_names}

    # Save state dict
    torch.save(filtered_state, target_file)
    print(f"[AGE] Saved {len(filtered_state)} UNet tensors to {target_file}")

def save_history(losses, name, word_print, models_path):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=7.5)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=200)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-4)
    parser.add_argument('--pretrained_model', help='Diffusers model id or checkpoint path (can be .ckpt/.safetensors)', type=str, required=False, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of erased_words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--save_freq', help='frequency to save data, per iteration', type=int, required=False, default=None)
    parser.add_argument('--models_path', help='Directory where fine-tuned checkpoints are stored', type=str, required=False, default='AGE_models')
    parser.add_argument('--save_path', help='Path (including filename) for filtered trainable UNet weights; defaults to trainable_unets/<run>_trainable_unet.pt', type=str, required=False, default=None)
    parser.add_argument('--dtype', help='Computation dtype for UNet/VAE (auto, float32, float16, bfloat16)', type=str, required=False, default='float32')
    parser.add_argument('--debug_concept', default=None, type=str, help='Debug concept to replace adaptive anchor during training')

    parser.add_argument('--gumbel_lr', help='learning rate for prompt', type=float, required=False, default=1e-2)
    parser.add_argument('--gumbel_temp', help='temperature for gumbel softmax', type=float, required=False, default=0.1)
    parser.add_argument('--gumbel_hard', help='hard for gumbel softmax, 0: soft, 1: hard', type=int, required=False, default=1, choices=[0,1])
    parser.add_argument('--gumbel_num_centers', help='number of centers for kmeans, if <= 0 then do not apply kmeans', type=int, required=False, default=100)
    parser.add_argument('--gumbel_update', help='update frequency for preserved set, if <= 0 then do not update', type=int, required=False, default=-1)
    parser.add_argument('--gumbel_time_step', help='time step for the starting point to estimate epsilon', type=int, required=False, default=0)
    parser.add_argument('--gumbel_multi_steps', help='multi steps for calculating the output', type=int, required=False, default=2)
    parser.add_argument('--gumbel_k_closest', help='number of closest tokens to consider', type=int, required=False, default=100)
    parser.add_argument('--gumbel_topk', help='number of top-k values in the soft gumbel softmax to be considered', type=int, required=False, default=2)
    parser.add_argument('--ignore_special_tokens', help='ignore special tokens in the embedding matrix', type=str2bool, required=False, default=True)
    parser.add_argument('--vocab', help='vocab', type=str, required=False, default='EN3K')
    parser.add_argument('--pgd_num_steps', help='number of step to optimize adversarial concepts', type=int, required=False, default=2)
    parser.add_argument('--lamda', help='lambda for the loss function', type=float, required=False, default=1)

    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    pretrained_model = args.pretrained_model
    devices = parse_device_list(args.devices)
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_age(
        prompt=prompt,
        train_method=train_method,
        start_guidance=start_guidance,
        negative_guidance=negative_guidance,
        iterations=iterations,
        lr=lr,
        pretrained_model=pretrained_model,
        devices=devices,
        seperator=seperator,
        image_size=image_size,
        ddim_steps=ddim_steps,
        args=args,
    )
