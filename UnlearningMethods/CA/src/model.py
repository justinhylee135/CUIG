# Standard Library Imports
from typing import Callable, Optional
import os

# Third Party Imports
import torch
from accelerate.logging import get_logger
from packaging import version

## Diffusers Imports
import diffusers
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import Attention as CrossAttention
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers import DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import (
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    PretrainedConfig,
    AutoTokenizer,
)

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = get_logger(__name__)

############################################### Directly Moved from train_ca.py
def create_custom_diffusion(unet, parameter_group):
    for name, params in unet.named_parameters():
        if parameter_group == "cross-attn":
            if "attn2.to_k" in name or "attn2.to_v" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "attn":
            if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                print(name)
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "full-weight":
            params.requires_grad = True
        elif parameter_group == "embedding":
            params.requires_grad = False
        else:
            raise ValueError(
                "parameter_group argument only cross-attn, full-weight, embedding"
            )

    # change attn class
    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(
                    layer, layer.__class__
                )
                setattr(
                    layer, "set_use_memory_efficient_attention_xformers", bound_method
                )
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


def save_model_card(
    repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"./image_{i}.png\n"

    yaml = f"""
        ---
        license: creativeml-openrail-m
        base_model: {base_model}
        instance_prompt: {prompt}
        tags:
        - stable-diffusion
        - stable-diffusion-diffusers
        - text-to-image
        - diffusers
        - custom diffusion
        inference: true
        ---
            """
    model_card = f"""
        # Custom Diffusion - {repo_id}

        These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
        {img_str[0]}
        """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        pass
        # from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
        #     RobertaSeriesModelWithTransformation,
        # )

        # return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def freeze_params(params):
    for param in params:
        param.requires_grad = False
###############################################
# Code chunks cut from main and moved here
def setup_models_and_tokenizer(args, accelerator):
    """
    Load and set up tokenizer, text encoder, VAE, UNet, and noise scheduler.
    
    Args:
        args: Arguments object containing model configuration
        accelerator: Accelerator object for mixed precision info
        
    Returns:
        tuple: (tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype)
    """
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.base_model_dir:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_dir,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    else:
        raise ValueError("Either tokenizer_name or base_model_dir must be provided")

    # Import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.base_model_dir, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_dir, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.base_model_dir,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.base_model_dir, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model_dir, subfolder="unet", revision=args.revision
    )

    # Set gradient requirements
    vae.requires_grad_(False)
    if args.parameter_group != "embedding":
        text_encoder.requires_grad_(False)
    unet = create_custom_diffusion(unet, args.parameter_group)

    # Determine weight dtype for mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype



# Original Functions for CA Repo Below
def set_use_memory_efficient_attention_xformers(
    self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
):
    if use_memory_efficient_attention_xformers:
        if self.added_kv_proj_dim is not None:
            # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
            # which uses this type of cross attention ONLY because the attention mask of format
            # [0, ..., -10.000, ..., 0, ...,] is not supported
            raise NotImplementedError(
                "Memory efficient attention with `xformers` is currently not supported when"
                " `self.added_kv_proj_dim` is defined."
            )
        elif not is_xformers_available():
            raise ModuleNotFoundError(
                (
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                    " xformers"
                ),
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                " only available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e

        processor = CustomDiffusionXFormersAttnProcessor(
            attention_op=attention_op)
    else:
        processor = CustomDiffusionAttnProcessor()

    self.set_processor(processor)


class CustomDiffusionAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if version.parse(diffusers.__version__) < version.parse("0.20.0"):
                if attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            else:
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CustomDiffusionXFormersAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if version.parse(diffusers.__version__) < version.parse("0.20.0"):
                if attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            else:
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg, StableDiffusionPipelineOutput

class CustomDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for custom diffusion model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker",
                            "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker)


    def save_pretrained(self, save_path, parameter_group="cross-attn", all=False):
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {'unet': {}}
            if parameter_group == 'embedding':
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
            for name, params in self.unet.named_parameters():
                if parameter_group == "cross-attn":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif parameter_group == "attn":
                    if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif parameter_group == "full-weight":
                    delta_dict['unet'][name] = params.cpu().clone()
                else:
                    raise ValueError(
                        "parameter_group argument only supports one of [cross-attn, full-weight, embedding]"
                    )
            torch.save(delta_dict, save_path)

    def load_model(self, save_path):
        st = torch.load(save_path)
        print(st.keys())
        if 'text_encoder' in st:
            self.text_encoder.load_state_dict(st['text_encoder'])
        for name, params in self.unet.named_parameters():
            if name in st['unet']:
                params.data.copy_(st['unet'][f'{name}'])