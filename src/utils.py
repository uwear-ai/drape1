import torch
from PIL import Image
import numpy as np
from einops import rearrange
import inspect
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, __version__
from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import IPAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import PeftAdapterMixin
from typing import  List, Optional, Union
from torchvision import transforms

from PIL import Image


class UwearLoaderMixin(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    IPAdapterMixin,
    UNet2DConditionLoadersMixin,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_single_file_uwear(
        self, model_path: str, config_path: str, torch_dtype=None
    ):
        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        config, unused_kwargs, commit_hash = self.load_config(
            config_path,
            cache_dir=None,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=False,
            proxies=None,
            local_files_only=None,
            token=None,
            revision=None,
            subfolder=None,
            user_agent=user_agent,
        )

        model = self.from_config(config, **unused_kwargs)

        state_dict = load_state_dict(model_path)
        model._convert_deprecated_attention_blocks(state_dict)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = (
            self._load_pretrained_model(
                model,
                state_dict,
                model_path,
                model_path,
                ignore_mismatched_sizes=False,
            )
        )

        if torch_dtype is not None:
            model.to(torch_dtype)

        model.eval()

        return model


def tokenize_prompt(tokenizer, prompt):
    text_input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_input_ids.input_ids
    return text_input_ids


def encode_prompt(text_encoder, tokenizer, prompt, get_pool=False):
    text_input_ids = tokenize_prompt(tokenizer, prompt)
    with torch.no_grad():
        prompt_embs = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True
        )
    if get_pool:
        pooled_prompt_embeds = prompt_embs[0]
    else:
        pooled_prompt_embeds = None

    prompt_embs = prompt_embs.hidden_states[-2]
    return prompt_embs, pooled_prompt_embeds


def encode_prompt_sdxl(
    text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, prompt
):
    prompt_embeds_list = []

    prompt_embed, _ = encode_prompt(text_encoder_1, tokenizer_1, prompt)
    prompt_embeds_list.append(prompt_embed)

    prompt_embed, pool = encode_prompt(
        text_encoder_2, tokenizer_2, prompt, get_pool=True
    )
    prompt_embeds_list.append(prompt_embed)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    return prompt_embeds, pool


def tensor_to_latents(
    vae: AutoencoderKL,
    image_tensors: torch.Tensor,
):
    latents = []
    image_tensors = image_tensors.to(vae.device, dtype=vae.dtype)
    for i in range(0, image_tensors.shape[0], 8):
        with torch.no_grad():
            latents.append(vae.encode(image_tensors[i : i + 8]).latent_dist.sample())
    latents = torch.cat(latents, dim=0)
    latents = latents * vae.config.scaling_factor
    return latents


def PIL_to_latents(vae: AutoencoderKL, image: Image):
    """
    Convert RGB image to VAE latents
    """
    image = [np.array(image).astype(np.float32) / 255.0]
    image = np.stack(image, axis=0)
    if image.ndim == 3:
        image = image[..., None]

    image_tensor = torch.from_numpy(image.transpose(0, 3, 1, 2))

    image_tensor = image_tensor.to(vae.device, dtype=vae.dtype)
    image_tensor = image_tensor * 2 - 1
    with torch.no_grad():
        latents = vae.encode(image_tensor)["latent_dist"].mean * vae.scaling_factor

    return latents


def comfy_image_to_latents(vae: AutoencoderKL, image: torch.tensor, dtype):
    image_tensor = rearrange(image, "b h w c -> b c h w")
    image_tensor = image_tensor.to(vae.device, dtype=vae.dtype)
    image_tensor = transforms.Normalize([0.5], [0.5])(image_tensor)
    with torch.no_grad():
        latent = vae.encode(image_tensor)["latent_dist"].mean * vae.scaling_factor

    return latent.to(dtype=vae.dtype)


def get_noisy_latents(
    num_channels_latents,
    height,
    width,
    scheduler,
    device="cuda",
    dtype=torch.float16,
    downscale_factor=8,
):

    shape = (
        1,
        num_channels_latents,
        height // downscale_factor,
        width // downscale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=dtype)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents


def latents_to_image(vae, latents):
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image.squeeze())
    return image


def prepare_image_controlnet(image: Image, height: int, width: int, device: str, dtype):
    processor = VaeImageProcessor(
        vae_scale_factor=8, do_convert_rgb=True, do_normalize=False
    )
    return processor.preprocess(image, height=height, width=width).to(
        device, dtype=dtype
    )


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    return timesteps


def sdxl_get_add_time_ids(
    unet,
    original_size,
    crops_coords_top_left,
    target_size,
    dtype,
    text_encoder_projection_dim=None,
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * len(add_time_ids)
        + text_encoder_projection_dim
    )
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def prepare_t2i_adapter_cond(image, resolution=(1024, 1024)):
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    if isinstance(image, Image.Image):
        return conditioning_image_transforms(image)
    elif isinstance(image, torch.Tensor):
        return image


def prepare_sdxl_params(
    prompt,
    negative_prompt,
    num_inference_steps,
    height,
    width,
    unet,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    scheduler,
    device,
    dtype,
    cfg_type="regular",
    do_classifier_free_guidance=True,
):
    params = {}
    params["timesteps"] = retrieve_timesteps(scheduler, num_inference_steps, device)
    params["latents"] = get_noisy_latents(
        unet.config.in_channels, height, width, scheduler, dtype=unet.dtype
    )
    prompt_embeds, prompt_pool = encode_prompt_sdxl(
        text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, prompt
    )
    if do_classifier_free_guidance:
        negative_prompt_embeds, negative_prompt_pool = encode_prompt_sdxl(
            text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, negative_prompt
        )

    original_size = target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = negative_add_time_ids = sdxl_get_add_time_ids(
        unet,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=text_encoder_2.config.projection_dim,
    )

    if cfg_type == "regular":
        params["prompt_embeds"] = (
            torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if do_classifier_free_guidance
            else prompt_embeds
        )
        add_text_embeds = (
            torch.cat([negative_prompt_pool, prompt_pool], dim=0).to(device)
            if do_classifier_free_guidance
            else prompt_pool.to(device)
        )
        add_time_ids = (
            torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(device)
            if do_classifier_free_guidance
            else add_time_ids.to(device)
        )
    else:
        params["prompt_embeds"] = torch.cat(
            [negative_prompt_embeds, prompt_embeds, negative_prompt_embeds], dim=0
        )
        add_text_embeds = torch.cat(
            [negative_prompt_pool, prompt_pool, negative_prompt_pool], dim=0
        ).to(device)
        add_time_ids = torch.cat(
            [negative_add_time_ids, add_time_ids, negative_add_time_ids], dim=0
        ).to(device)

    params["added_cond_kwargs"] = {
        "text_embeds": add_text_embeds,
        "time_ids": add_time_ids,
    }

    return params


def image_to_latents(vae, image):
    if isinstance(image, Image.Image):
        ref_latents = PIL_to_latents(vae, image)
    elif isinstance(image, torch.Tensor):
        assert (
            len(image.shape) == 4
        ), f"Image tensor should have 4 dimension, got {len(image.shape)}"
        ref_latents = tensor_to_latents(vae, image)

    return ref_latents


def prepare_ref_ma_uwear_cond(
    vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, image_ref, prompt_ref
):

    ref_latents = image_to_latents(vae, image_ref)
    ref_embeds, ref_pool = encode_prompt_sdxl(
        text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, prompt_ref
    )

    return ref_latents, ref_embeds, ref_pool


def prepare_ref_ip_embedding(image_encoder, image_proj_model, processed_image):
    device = image_encoder.device
    dtype = image_encoder.dtype
    clip_image_embeds = image_encoder(
        processed_image.to(device, dtype=dtype), output_hidden_states=True
    ).hidden_states[-2]
    neg_clip_image_embeds = image_encoder(
        torch.zeros_like(processed_image).to(device, dtype=dtype),
        output_hidden_states=True,
    ).hidden_states[-2]
    image_proj = image_proj_model(clip_image_embeds)
    neg_image_proj = image_proj_model(neg_clip_image_embeds)
    return image_proj, neg_image_proj


def get_t5_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def get_clip_prompt_embeds_flux(
    text_encoder,
    tokenizer,
    tokenizer_max_length,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt_flux(
    text_encoder_clip,
    text_encoder_t5,
    tokenizer_clip,
    tokenizer_t5,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
):
    dtype = text_encoder_clip.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = get_clip_prompt_embeds_flux(
            text_encoder=text_encoder_clip,
            tokenizer=tokenizer_clip,
            tokenizer_max_length=tokenizer_clip.model_max_length,
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = get_t5_prompt_embeds(
            text_encoder=text_encoder_t5,
            tokenizer=tokenizer_t5,
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=tokenizer_t5.model_max_length,
            device=device,
        )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def flux_prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def prepare_latents_flux(
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    latents=None,
    vae_scale_factor=16,
):
    height = 2 * (int(height) // vae_scale_factor)
    width = 2 * (int(width) // vae_scale_factor)

    shape = (batch_size, num_channels_latents, height, width)

    if latents is not None:
        latent_image_ids = flux_prepare_latent_image_ids(
            batch_size, height, width, device, dtype
        )
        return latents.to(device=device, dtype=dtype), latent_image_ids

    latents = torch.randn(shape, device=device, dtype=dtype)
    latents = flux_pack_latents(
        latents, batch_size, num_channels_latents, height, width
    )

    latent_image_ids = flux_prepare_latent_image_ids(
        batch_size, height, width, device, dtype
    )

    return latents, latent_image_ids


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps_flux(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def prepare_timesteps_flux(scheduler, num_inference_steps, latents, device):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps_flux(
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    return timesteps, num_warmup_steps


def segment_and_resize(image: Image, remover, height: int = 1024, width: int = 768):
    # Step 1: Segment the image using transparent-background
    segmented_image = remover.process(image)

    # Step 2: Determine the bounding box of the non-transparent region
    bbox = segmented_image.getbbox()
    if not bbox:
        raise ValueError("The image does not contain any non-transparent pixels.")

    # Crop the image to the bounding box
    cropped_image = segmented_image.crop(bbox)

    # Step 3: Add a 50-pixel border around the cropped image
    border_size = 50
    width_with_border = cropped_image.width + 2 * border_size
    height_with_border = cropped_image.height + 2 * border_size

    # Calculate the maximum size while keeping the border and maintaining aspect ratio
    target_size = (width - 2 * border_size, height - 2 * border_size)
    scale = min(
        target_size[0] / cropped_image.width, target_size[1] / cropped_image.height
    )

    # Resize the image using the calculated scale
    new_size = (int(cropped_image.width * scale), int(cropped_image.height * scale))
    resized_image = cropped_image.resize(new_size, Image.LANCZOS)

    # Step 4: Create a new white background of size 768x1024
    final_size = (width, height)
    background = Image.new("RGBA", final_size, (255, 255, 255, 255))

    # Step 5: Calculate the position to paste the resized image onto the background
    position = (
        (final_size[0] - resized_image.width) // 2,  # Center horizontally
        (final_size[1] - resized_image.height) // 2,  # Center vertically
    )

    # Step 6: Paste the resized image onto the background
    background.paste(resized_image, position, resized_image)

    # Convert to RGB mode if needed and return the final image
    return background.convert("RGB")


def prepare_prompt(prompt: str) -> str:
    out_prompt = "a person wearing clothes, " + prompt + ", photography"
    return out_prompt