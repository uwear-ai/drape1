import gc
import os
from diffusers.image_processor import VaeImageProcessor
from .utils import (
    prepare_sdxl_params,
    prepare_ref_ma_uwear_cond,
    latents_to_image,
    segment_and_resize,
    prepare_prompt,
)
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from .unet_ref import UwearUNet2DConditionModelRef
from .unet_base import UwearUNet2DConditionModel
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from random import randint

def sample_uwear_parallel_unets(
    unet,
    unet_ref,
    scheduler,
    timesteps,
    guidance_scale,
    latents,
    ref_latents,
    prompt_embeds,
    ref_embeds,
    added_cond_kwargs,
    ref_added_cond_kwargs,
    low_vram=True,
):
    if low_vram:
        unet_ref.to("cuda")

    with torch.no_grad():
        hidden_states_refs = unet_ref(
            ref_latents,
            torch.Tensor([0]).to("cuda"),
            encoder_hidden_states=ref_embeds,
            added_cond_kwargs=ref_added_cond_kwargs,
            return_dict=False,
        )

    if low_vram:
        unet_ref.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    for t in tqdm(timesteps, position=0, leave=True):

        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                hidden_states_ref=hidden_states_refs,
            )

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


class DrapeDiffusionPipeline:
    def __init__(
        self,
        unet,
        unet_ref,
        vae: AutoencoderKL,
        text_encoder_1: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_1: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        scheduler,
        t2i_adapter=None,
        device="cuda",
        dtype=torch.float16,
    ):
        self.unet = unet
        self.unet_ref = unet_ref
        self.vae = vae
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.t2i_adapter = t2i_adapter
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.control_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def __call__(
        self,
        prompt: str,
        image_ref: Image,
        num_inference_steps: int = 30,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        prompt_ref: str = "",
        seed: int = None,
        low_vram: bool = True,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        if low_vram:
            self.unet_ref.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self.vae.to("cuda")
            self.text_encoder_1.to("cuda")
            self.text_encoder_2.to("cuda")
            
        params = prepare_sdxl_params(
            prompt,
            negative_prompt,
            num_inference_steps,
            height,
            width,
            self.unet,
            self.text_encoder_1,
            self.text_encoder_2,
            self.tokenizer_1,
            self.tokenizer_2,
            self.scheduler,
            self.device,
            self.dtype,
        )
        image_ref = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )(image_ref).unsqueeze(0)

        ref_latents, ref_embeds, ref_pool = prepare_ref_ma_uwear_cond(
            self.vae,
            self.text_encoder_1,
            self.text_encoder_2,
            self.tokenizer_1,
            self.tokenizer_2,
            image_ref,
            prompt_ref,
        )

        if low_vram:
            self.vae.to("cpu")
            self.text_encoder_1.to("cpu")
            self.text_encoder_2.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        ref_latents = torch.cat([torch.zeros_like(ref_latents), ref_latents])

        ref_embeds = torch.cat([torch.zeros_like(ref_embeds), ref_embeds])

        ref_pool = torch.cat([torch.zeros_like(ref_pool), ref_pool])

        ref_added_cond_kwargs = {
            "time_ids": params["added_cond_kwargs"]["time_ids"],
            "text_embeds": ref_pool,
        }

        latents = sample_uwear_parallel_unets(
            unet=self.unet,
            unet_ref=self.unet_ref,
            scheduler=self.scheduler,
            timesteps=params["timesteps"],
            guidance_scale=guidance_scale,
            latents=params["latents"],
            ref_latents=ref_latents,
            prompt_embeds=params["prompt_embeds"],
            ref_embeds=ref_embeds,
            added_cond_kwargs=params["added_cond_kwargs"],
            ref_added_cond_kwargs=ref_added_cond_kwargs,
            low_vram=low_vram,
        )
        if low_vram:
            self.vae.to("cuda")
        out = latents_to_image(self.vae, latents)
        if low_vram:
            self.vae.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        return out


def load_drape(low_vram:bool=True):

    model_repo_id = "Uwear-ai/Drape1" 
    torch_dtype = torch.float16
    device = "cuda"

    unet = UwearUNet2DConditionModel.from_pretrained(
        model_repo_id,
        subfolder="unet",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        device_map=None,
        token=os.getenv("TOKEN_HF"),
    ).to(device)

    unet_ref = UwearUNet2DConditionModelRef.from_pretrained(
        model_repo_id,
        subfolder="unet_ref",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    if not low_vram:
        unet_ref.to(device)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
    )
    if not low_vram:
        vae.to(device)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_repo_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    )
    if not low_vram:
        text_encoder_1.to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_repo_id,
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
    )
    if not low_vram:
        text_encoder_2.to(device)
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        model_repo_id, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_repo_id, subfolder="tokenizer_2"
    )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        model_repo_id, subfolder="scheduler"
    )

    pipe = DrapeDiffusionPipeline(
        unet=unet,
        unet_ref=unet_ref,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        device=device,
        dtype=torch_dtype,
    )
    return pipe


def infer_drape(
    pipe: DrapeDiffusionPipeline,
    prompt: str, 
    image_ref: Image, 
    remover, 
    seed: int = None,
    width: int = 768,
    height: int = 1024,
    guidance_scale: float = 2,
    num_inference_steps: int = 20,
    prompt_ref: str = "clothes",
    negative_prompt: str = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
):

    prompt = prepare_prompt(prompt=prompt)
    image_ref = segment_and_resize(image=image_ref, remover=remover, width=width, height=height)
    if not seed:
        seed = randint(0, 1000000)
    image = pipe(
        prompt=prompt,
        image_ref=image_ref,
        prompt_ref=prompt_ref,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        seed=seed,
    )
    return image