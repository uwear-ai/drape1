
# Modified from the original source code
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from .utils import UwearLoaderMixin

from diffusers.models.attention_processor import Attention, AttnProcessor2_0

"""def prepare_mask(mask, q_size, latent_witdh, latent_height):
    mask_h = latent_height / math.sqrt(latent_height * latent_witdh / q_size)
    mask_h = int(mask_h) + int((q_size % int(mask_h)) != 0)
    mask_w = q_size // mask_h
    mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
    mask_downsample = mask_downsample.repeat(2, 1, 1)
    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1, 1).repeat(1, 1, out.shape[2])
    opposite_mask_downsample = 
    return mask_downsample, opposite_mask_downsample
"""
# SDXL
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"


class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


'''class Attention(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)

        if encoder_hidden_states is not None:
            to_k_v = encoder_hidden_states
        else:
            q = self.to_q(hidden_states)
            to_k_v = hidden_states

        k = self.to_k(to_k_v)
        v = self.to_v(to_k_v)
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output'''


class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size,use_ref:bool):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.use_ref = use_ref
        self.attn1 = Attention(hidden_size,heads=hidden_size // 64,processor=AttnProcessor2_0())
        if self.use_ref:
            self.attn1_ref = Attention(hidden_size,heads=hidden_size // 64,processor=AttnProcessor2_0())
            
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 2048,heads=hidden_size // 64,processor=AttnProcessor2_0())
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None, hidden_states_ref=None, mask=None,weight_base:float=1,weight_ref:float=1):

        residual = x
        x_norm = self.norm1(x)

        if hidden_states_ref is None:
            out_hidden_states_ref = x
        else:
            x_ref = torch.cat([x,hidden_states_ref.to(dtype=x.dtype)],dim=1)
            x_ref = self.norm1(x_ref)
            x_ref = self.attn1_ref(x_ref)
            x_ref, _ = x_ref.chunk(2, dim=1)
            out_hidden_states_ref = None

        x = self.attn1(x_norm)
        x = x * weight_base

        if hidden_states_ref is not None:
            if self.use_ref:     

                x = x + x_ref*weight_ref

        x = x + residual

        residual = x

        x = self.norm2(x)
        x = self.attn2(x, encoder_hidden_states)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x, out_hidden_states_ref


class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers,use_ref):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels,use_ref) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        hidden_states_ref=None,
        mask=None,
        weight_base:float=1,
        weight_ref:float=1
    ):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        out_hidden_states_refs = []
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, out_hidden_states_ref = block(
                hidden_states,
                encoder_hidden_states,
                None if hidden_states_ref is None else hidden_states_ref[i],
                mask=mask,
                weight_base=weight_base,
                weight_ref=weight_ref
            )
            out_hidden_states_refs.append(out_hidden_states_ref)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res, out_hidden_states_refs


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=False),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = self.downsamplers[0](hidden_states)
        output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True,use_ref=True):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers,use_ref),
                Transformer2DModel(out_channels, out_channels, n_layers,use_ref),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

    def forward(
        self,
        hidden_states,
        temb,
        encoder_hidden_states,
        hidden_states_ref=None,
        additional_residuals=None,
        mask=None,
    ):
        output_states = []
        out_hidden_states_refs = []
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            hidden_states = resnet(hidden_states, temb)
            hidden_states, out_hidden_states_ref = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                hidden_states_ref=(
                    None if hidden_states_ref is None else hidden_states_ref[i]
                ),
                mask=mask,
            )
            if i == len(self.attentions) - 1 and additional_residuals is not None:
                hidden_states += additional_residuals
            output_states.append(hidden_states)
            out_hidden_states_refs.append(out_hidden_states_ref)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states, out_hidden_states_refs


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers,use_ref):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers,use_ref),
                Transformer2DModel(out_channels, out_channels, n_layers,use_ref),
                Transformer2DModel(out_channels, out_channels, n_layers,use_ref),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(2 * out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb,
        encoder_hidden_states,
        hidden_states_ref=None,
        mask=None,
        weight_base:float=1,
        weight_ref:float=1
    ):
        out_hidden_states_refs = []
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states, out_hidden_states_ref = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                hidden_states_ref=(
                    None if hidden_states_ref is None else hidden_states_ref[i]
                ),
                mask=mask,
                weight_base=weight_base,
                weight_ref=weight_ref
            )
            out_hidden_states_refs.append(out_hidden_states_ref)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, out_hidden_states_refs


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels * 2, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features,use_ref):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [Transformer2DModel(in_features, in_features, n_layers=10,use_ref=use_ref)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        hidden_states_ref=None,
        additional_residuals=None,
        mask=None,
        weight_base:float=1,
        weight_ref:float=1
    ):
        out_hidden_states_refs = []
        hidden_states = self.resnets[0](hidden_states, temb)
        for i, (attn, resnet) in enumerate(zip(self.attentions, self.resnets[1:])):
            hidden_states, out_hidden_states_ref = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                hidden_states_ref=(
                    None if hidden_states_ref is None else hidden_states_ref[i]
                ),
                mask=mask,
                weight_base=weight_base,
                weight_ref=weight_ref
            )
            if i == len(self.attentions) - 1 and additional_residuals is not None:
                hidden_states += additional_residuals
            out_hidden_states_refs.append(out_hidden_states_ref)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states, out_hidden_states_refs


class UwearUNet2DConditionModelBase(nn.Module):
    def __init__(self):
        super(UwearUNet2DConditionModelBase, self).__init__()

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=320, out_channels=320),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2,use_ref=False),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                    use_ref=False
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                    use_ref=True
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                    use_ref=True
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = UNetMidBlock2DCrossAttn(1280,use_ref=True)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

        
    @property
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def save_uwear_adapter(self,folder_path):
        from copy import deepcopy
        from safetensors.torch import save_file
        attn = {}
        for name, param in self.named_parameters():
            if 'attn1_ref.' in name:
                attn[name]=deepcopy(param)
        save_file(attn,os.path.join(folder_path,"uwear_adapter.safetensors"))


    def load_uwear_adapter(self,uwear_adapter_path):
        from safetensors import safe_open
        from torch.nn import Parameter
        attn = {}
        state_dict = {}
        with safe_open(uwear_adapter_path, framework="pt", device=0) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        for name, param in self.named_parameters():
            if 'attn1_ref.' in name:
                attn[name]=state_dict[name]

        def set_nested_attr(model, attr, value):
            attrs = attr.split('.')
            for a in attrs[:-1]:
                model = getattr(model, a)
            # Convert the value to torch.nn.Parameter if it isn't already
            if isinstance(value, torch.Tensor) and not isinstance(value, Parameter):
                value = Parameter(value)
            setattr(model, attrs[-1], value)
            
        for layer_path,param in attn.items():
            set_nested_attr(self, layer_path, param)

        self.to(self.device,dtype=self.dtype)


    def forward(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        added_cond_kwargs,
        hidden_states_ref=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        mask=None,
        weight_base:float=1,
        weight_ref:float=1,
        **kwargs
    ):
        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = (
            mid_block_additional_residual is None
            and down_block_additional_residuals is not None
        )
        sample = sample.to(self.dtype)
        out_hidden_states_refs = dict()
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds").to(dtype=sample.dtype)
        time_ids = added_cond_kwargs.get("time_ids").to(dtype=sample.dtype)

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(sample.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)
        num_adapter_res = 0
        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )
        if is_adapter:
            sample += down_block_additional_residuals[num_adapter_res]
            num_adapter_res += 1

        down_add_res = (
            down_block_additional_residuals[num_adapter_res] if is_adapter else None
        )
        num_adapter_res += 1

        sample, [s4, s5, s6], out_hidden_states_ref = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            additional_residuals=down_add_res,
            mask=mask,
        )
        out_hidden_states_refs["down_1"] = out_hidden_states_ref

        down_add_res = (
            down_block_additional_residuals[num_adapter_res] if is_adapter else None
        )
        num_adapter_res += 1
        sample, [s7, s8], out_hidden_states_ref = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            additional_residuals=down_add_res,
            mask=mask,
        )
        out_hidden_states_refs["down_2"] = out_hidden_states_ref
        down_add_res = (
            down_block_additional_residuals[num_adapter_res] if is_adapter else None
        )

        if is_controlnet:
            s0 = s0 + down_block_additional_residuals[0]
            s1 = s1 + down_block_additional_residuals[1]
            s2 = s2 + down_block_additional_residuals[2]
            s3 = s3 + down_block_additional_residuals[3]
            s4 = s4 + down_block_additional_residuals[4]
            s5 = s5 + down_block_additional_residuals[5]
            s6 = s6 + down_block_additional_residuals[6]
            s7 = s7 + down_block_additional_residuals[7]
            s8 = s8 + down_block_additional_residuals[8]

        # 4. mid
        sample, out_hidden_states_ref = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_ref=(
                None if hidden_states_ref is None else hidden_states_ref["mid"]
            ),
            additional_residuals=down_add_res,
            mask=mask,
            weight_base=weight_base,
            weight_ref=weight_ref
        )

        if is_controlnet:
            sample += mid_block_additional_residual

        out_hidden_states_refs["mid"] = out_hidden_states_ref

        # 5. up
        sample, out_hidden_states_ref = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_ref=(
                None if hidden_states_ref is None else hidden_states_ref["up_0"]
            ),
            mask=mask,
            weight_base=weight_base,
            weight_ref=weight_ref
        )
        out_hidden_states_refs["up_0"] = out_hidden_states_ref

        sample, out_hidden_states_ref = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_ref=(
                None if hidden_states_ref is None else hidden_states_ref["up_1"]
            ),
            mask=mask,
            weight_base=weight_base,
            weight_ref=weight_ref
        )
        out_hidden_states_refs["up_1"] = out_hidden_states_ref
        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class UwearUNet2DConditionModel(
    UwearUNet2DConditionModelBase,UwearLoaderMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass