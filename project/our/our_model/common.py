import torch
from torch import nn

from mmdet.registry import MODELS

from mmengine.model import BaseModule
from mmengine.dist import is_main_process

from peft import get_peft_config, get_peft_model

from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from .sam import UAViTEncoder


class ColorAttentionAdapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, change=False) -> None:
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, embedding_dim, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.change_channel = change

    def forward(self, x):
        if self.change_channel:
            x = x.permute(0, 3, 1, 2).contiguous()
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
            return self.Sigmoid(avg_out + max_out).view(x.shape[0], 1, 1, -1)
        else:
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
            return self.Sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)


class Adapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, skip=False, scale=1):
        super().__init__()
        self.skip = skip
        self.scale = scale
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.act = act_layer()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        if self.skip:
            out = out + x
        return self.scale * out


class UAViTBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 use_color_adapter=True,
                 use_space_adapter=True,
                 use_mlp_adapter=True,
                 ):
        super().__init__()

        if use_color_adapter:
            self.color_adapter = ColorAttentionAdapter(embed_dim, change=True)
        if use_space_adapter:
            self.space_adapter = Adapter(embed_dim, skip=True)
        if use_mlp_adapter:
            self.mlp_adapter = Adapter(embed_dim, scale=0.5)


@MODELS.register_module()
class UAViTAdapters(BaseModule):

    def __init__(self,
                 adapter_layer,
                 embed_dim,
                 use_color_adapter=True,
                 use_space_adapter=True,
                 use_mlp_adapter=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.adapter_layer = adapter_layer
        for idx in adapter_layer:
            self.add_module(
                f'adapter_{idx}',
                UAViTBlock(embed_dim, use_color_adapter, use_space_adapter, use_mlp_adapter)
            )


class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.GELU) -> None:
        super().__init__()
        self.act = act_layer()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv3 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(output_dim, output_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(output_dim, output_dim, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv3(x) + self.conv5(x) + self.conv7(x)
        return self.act(self.bn2(x))


@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class USISSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class USISSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class USISSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = UAViTEncoder(sam_config)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class USISSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)
