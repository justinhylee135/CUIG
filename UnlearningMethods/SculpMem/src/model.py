# Standard Library Imports
from typing import Callable, Optional
import os

# Third Party Imports
import torch
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
    CLIPTextConfig,
    AutoTokenizer,
)

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

############################################### 
# Portions shared with the ConAbl baseline.
def create_custom_unet(unet, parameter_group):
    # Set trainable parameters
    print(f"Setting trainable parameter group '{parameter_group}'")
    for name, params in unet.named_parameters():
        if parameter_group == "kv-xattn": # This is the default choice
            if "attn2.to_k" in name or "attn2.to_v" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "xattn":
            if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "full":
            params.requires_grad = True
        elif parameter_group == "text-emb":
            params.requires_grad = False
        else:
            raise ValueError(
                "parameter_group argument only cross-attn, full-weight, embedding"
            )

    # Calculate number of trainable parameters
    num_trainable = 0
    for name, params in unet.named_parameters():
        if params.requires_grad:
            num_trainable += params.numel()
    print(f"'{parameter_group}': Number of trainable parameters: '{num_trainable:,}'")

    # Use memory efficient cross-attention layers
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


def write_model_card_to_output_dir(
    repo_id: str, base_model_name=str, anchor_target_concepts=list, repo_folder=None
):

    yaml = f"""
        ---
        License: creativeml-openrail-m
        Base Model Name: {base_model_name}
        Unlearned Target to Anchor Mappings (anchor+target): {anchor_target_concepts}
        ---
            """
    model_card = f"""
        # Unlearned Diffusion Model - {repo_id}

        These are Diffusion weights for {base_model_name} after unlearning the target concepts.
        """
    
    # Store yaml and model card to output_dir
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    # Load CLIP Config
    text_encoder_config = CLIPTextConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"'{model_class}' is not supported.")


def freeze_params(params):
    for param in params:
        param.requires_grad = False

###############################################
# Model setup helpers shared with ConAbl.
def setup_models_and_tokenizer(args, accelerator):
    """
    Load and set up tokenizer, text encoder, VAE, UNet, and noise scheduler.
    
    Args:
        args: Arguments object containing model configuration
        accelerator: Accelerator object for mixed precision info
        
    Returns:
        tuple: (tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype)
    """
    
    print(f"Loading model components...")

    # Load tokenizer
    if args.tokenizer_name: # If you want to explicitly choose a different tokenizer
        print(f"\tLoading tokenizer from '{args.tokenizer_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.hf_revision,
            use_fast=False,
            clean_up_tokenization_spaces=True
        )
    elif args.base_model_dir:
        print(f"\tLoading tokenizer from '{args.base_model_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_dir,
            subfolder="tokenizer",
            revision=args.hf_revision,
            use_fast=False,
            clean_up_tokenization_spaces=True
        )
    else:
        raise ValueError("Either tokenizer_name or base_model_dir must be provided")

    # Text Encoder
    print(f"\tLoading text encoder from '{args.base_model_dir}'...")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model_dir, args.hf_revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.base_model_dir,
        subfolder="text_encoder",
        revision=args.hf_revision,
    )

    # Noise scheduler
    print(f"\tLoading noise scheduler from '{args.base_model_dir}'...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_dir, subfolder="scheduler"
    )

    # Image Encoder
    print(f"\tLoading VAE from '{args.base_model_dir}'...")
    vae = AutoencoderKL.from_pretrained(
        args.base_model_dir, subfolder="vae", revision=args.hf_revision, allow_pickle=True
    )

    # Denoising UNet
    print(f"\tLoading UNet from '{args.base_model_dir}'...")
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model_dir, subfolder="unet", revision=args.hf_revision
    )

    # Load UNet checkpoint (Previously Unlearned Model) for continual unlearning
    if args.unet_ckpt:
        if not os.path.exists(args.unet_ckpt):
            print(f"UNet checkpoint not found at '{args.unet_ckpt}'. Using default UNet from pipeline '{args.base_model_dir}'...")
        else:
            print(f"Loading UNet checkpoint from '{args.unet_ckpt}'...")
            ckpt_sd = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)
            if "unet" in ckpt_sd: ckpt_sd = ckpt_sd["unet"]            
            missing, unexpected = unet.load_state_dict(ckpt_sd, strict=False)
            print(f"\tLoaded '{len(ckpt_sd)}' keys from UNet checkpoint with '{len(missing)}' missing and '{len(unexpected)}' unexpected keys.")

    # Set gradient tracking requirements
    print(f"Setting parameter gradient tracking...")
    vae.requires_grad_(False)
    if args.parameter_group != "embedding":
        text_encoder.requires_grad_(False)
    
    # Set trainable parameters and replace cross attention with xformers memory efficient version
    unet = create_custom_unet(unet, args.parameter_group)

    # Determine weight dtype for mixed precision (Default: float32)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"Using weight dtype: '{weight_dtype}'")

    return tokenizer, text_encoder, vae, unet, noise_scheduler, weight_dtype



# Original Concept Ablation save/load helpers below.
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
    """Attention processor for custom diffusion that implements selective gradient detachment."""
    
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        # Get batch size and sequence length from hidden states
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Prepare attention mask for the given sequence length and batch size
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        
        # Project hidden states to query representation
        query = attn.to_q(hidden_states)

        # Flag to track if this is cross-attention (between different modalities)
        crossattn = False
        if encoder_hidden_states is None:
            # Self-attention: use hidden states as encoder input
            encoder_hidden_states = hidden_states
        else:
            # Cross-attention: normalize encoder hidden states based on diffusers version
            crossattn = True
            if version.parse(diffusers.__version__) < version.parse("0.20.0"):
                if attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            else:
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Project encoder hidden states to key and value representations
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # For cross-attention, selectively detach gradients (except for first token)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.  # Keep first token attached
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        # Reshape query, key, value from (batch, seq, heads*dim) to (batch*heads, seq, dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention weights and apply softmax
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Apply attention weights to values
        hidden_states = torch.bmm(attention_probs, value)
        
        # Reshape back to (batch, seq, heads*dim)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Apply output projection layers
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        return hidden_states


class CustomDiffusionXFormersAttnProcessor:
    """Attention processor for custom diffusion using xformers memory-efficient attention."""
    
    def __init__(self, attention_op: Optional[Callable] = None):
        """
        Initialize the xformers attention processor.
        
        Args:
            attention_op: Optional attention operation to use with xformers
        """
        self.attention_op = attention_op

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        Forward pass for memory-efficient attention computation.
        
        Args:
            attn: CrossAttention layer to process
            hidden_states: Input hidden states for query projection
            encoder_hidden_states: Optional encoder states for key/value projection
            attention_mask: Optional mask for attention computation
            
        Returns:
            Processed hidden states after attention
        """
        # Get batch size and sequence length from hidden states
        batch_size, sequence_length, _ = hidden_states.shape

        # Prepare attention mask for the given sequence length and batch size
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        # Project hidden states to query representation
        query = attn.to_q(hidden_states)

        # Flag to track if this is cross-attention (between different modalities)
        crossattn = False
        if encoder_hidden_states is None:
            # Self-attention: use hidden states as encoder input
            encoder_hidden_states = hidden_states
        else:
            # Cross-attention: normalize encoder hidden states based on diffusers version
            crossattn = True
            if version.parse(diffusers.__version__) < version.parse("0.20.0"):
                if attn.cross_attention_norm:
                    encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            else:
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Project encoder hidden states to key and value representations
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # For cross-attention, selectively detach gradients (except for first token)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.  # Keep first token attached
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        # Reshape and make contiguous for xformers: (batch, seq, heads*dim) to (batch*heads, seq, dim)
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # Compute memory-efficient attention using xformers
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        
        # Convert output to query dtype
        hidden_states = hidden_states.to(query.dtype)
        
        # Reshape back from (batch*heads, seq, dim) to (batch, seq, heads*dim)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Apply output projection layers
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout
        
        return hidden_states

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
    # Components that are optional and may be None
    _optional_components = ["safety_checker", "feature_extractor"]

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
        # Initialize parent StableDiffusionPipeline with all components
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker)

    def save_pretrained(self, save_path, parameter_group="kv-xattn", all=False):
        """
        Save model weights to local, optionally saving only trainable parameters.
        
        Args:
            save_path: Path where model will be saved
            parameter_group: Which parameters to save - "kv-xattn", "xattn", "full", or "embedding"
            all: If True, save entire pipeline; if False, save only delta weights
        """
        # If all=True, use parent class method to save full pipeline
        if all:
            super().save_pretrained(save_path)
        else:
            # Initialize dictionary to store only the parameters that have been changed
            delta_dict = {'unet': {}}
            
            # If training embeddings, also save text encoder weights
            if parameter_group == 'text-emb':
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
            
            # Iterate through UNet parameters and save based on parameter_group
            for name, params in self.unet.named_parameters():
                
                # Save only cross-attention key and value projection weights
                if parameter_group == "kv-xattn":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()

                elif parameter_group == "xattn": # Save all attention projection weights (query, key, value, output)
                    if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                        delta_dict['unet'][name] = params.cpu().clone()

                elif parameter_group == "full": # Save all UNet parameters
                    delta_dict['unet'][name] = params.cpu().clone()

                else: # Uknown parameter group
                    raise ValueError(f"Parameter group '{parameter_group}' unrecognized. Code only supports '[kv-xattn, xattn, full, text-emb]'")
            
            # Save the updated parameters to the save path
            torch.save(delta_dict, save_path)

    def load_model(self, save_path):
        """
        Load model weights from local and apply them to the current model.
        
        Args:
            save_path: Path to saved model weights
        """
        # Load the diffusion state dictionary
        print(f"Loading state dictionary from '{save_path}'")
        diffusion_state_dict = torch.load(save_path)

        # Print status
        print(f"Found '{len(diffusion_state_dict.keys())}' keys from '{diffusion_state_dict}' state dictionary. Keys are: '{diffusion_state_dict.keys()}'")
        
        # Load text encoder if it was saved
        if 'text_encoder' in diffusion_state_dict:
            print(f"Loading text_encoder from '{save_path}'")
            self.text_encoder.load_diffusion_state_dictate_dict(diffusion_state_dict['text_encoder'])
        
        # Load UNet parameters - iterate through current UNet and update weights that were saved
        num_updated = 0
        for name, params in self.unet.named_parameters():
            if name in diffusion_state_dict['unet']:
                num_updated += 1
                params.data.copy_(diffusion_state_dict['unet'][f'{name}'])
        print(f"Updated '{num_updated}' keys in UNet using state dictionary from '{save_path}'")
