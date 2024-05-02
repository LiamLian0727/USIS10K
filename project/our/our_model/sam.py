import torch.utils.checkpoint
from torch import nn
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoderOutput, SamVisionLayer, SamPatchEmbeddings, SamVisionNeck
)
from typing import Optional, Tuple, Union


class UAViTLayer(SamVisionLayer):

    @torch.no_grad()
    def layer_norm1_no_grad(self, x):
        return self.layer_norm1(x)

    @torch.no_grad()
    def layer_norm2_no_grad(self, x):
        return self.layer_norm2(x)

    @torch.no_grad()
    def attn_no_grad(self, hidden_states, output_attentions=False):
        return self.attn(hidden_states, output_attentions)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            adapter: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1_no_grad(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn_no_grad(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )

        if getattr(adapter, 'space_adapter', False):
            hidden_states = adapter.space_adapter(hidden_states)

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        if getattr(adapter, 'color_adapter', False):
            hidden_states = hidden_states * adapter.color_adapter(hidden_states)
        hidden_states = residual + hidden_states

        layernorm_output = self.layer_norm2_no_grad(hidden_states)

        if getattr(adapter, 'mlp_adapter', False):
            hidden_states = hidden_states + self.mlp(layernorm_output) + adapter.mlp_adapter(hidden_states)
        else:
            hidden_states = hidden_states + self.mlp(layernorm_output)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class UAViTEncoder(nn.Module):

    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = UAViTLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    @torch.no_grad()
    def patch_embed_no_grad(self, x):
        return self.patch_embed(x)

    @torch.enable_grad()
    def patch_embed_grad(self, x):
        return self.patch_embed(x)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            adapter: Optional[torch.nn.Module] = None,
            patch_embed_grad: Optional[bool] = False,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if patch_embed_grad:
            hidden_states = self.patch_embed_grad(pixel_values)
        else:
            hidden_states = self.patch_embed_no_grad(pixel_values)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions,
                    adapter=getattr(adapter, f'adapter_{i}', None)
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
