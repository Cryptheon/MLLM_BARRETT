import logging
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
from functools import partial
import math
import torch
from typing import Optional, Tuple, List, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (LlamaConfig, 
                          LlamaForCausalLM, 
                          LlamaModel,
                          LlamaPreTrainedModel)

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (is_torch_flex_attn_available, 
                                add_start_docstrings, 
                                can_return_tuple)

from transformers.models.llama.modeling_llama import (LlamaDecoderLayer, 
                                                      LlamaAttention,
                                                      LlamaMLP, 
                                                      LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv,
                                                      eager_attention_forward)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

BaseModelOutputWithPast


IGNORE_INDEX = -100
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# from transformers we edit the original LlamaDecoderLayer 
# to account for cross attention
# https://github.com/ArtificialZeng/transformers-Explained/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        wsi_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # use cross attention
        if wsi_embeds is not None:
            wsi_shape = wsi_embeds.shape[:-1]
            wsi_shape =  (*wsi_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(wsi_embeds).view(wsi_shape).transpose(1, 2)
            value_states = self.v_proj(wsi_embeds).view(wsi_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # the apply_rotary_pos_emb function broadcasts to the text sequence length
        # correct the length of the key states, which should be the same as the value states
        key_states = key_states[:,:,:value_states.shape[2],:]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class CrossAttentionLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        wsi_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            wsi_embeds = wsi_embeds,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class CrossAttentionLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, embed_wsi: bool = False, wsi_num_hidden_layers: int = None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if not embed_wsi:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers if wsi_num_hidden_layers is None else wsi_num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        wsi_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                # Inject cross-attn inputs only to the modified layer
                if isinstance(decoder_layer, CrossAttentionLlamaDecoderLayer):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        wsi_embeds=wsi_embeds,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class PathoLlamaConfig(LlamaConfig):
    model_type = "patho_llama"
    training = True

class PathoLlamaForCausalLM(LlamaForCausalLM):
    config_class = PathoLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = CrossAttentionLlamaModel(config)
        self.wsi_embedding_layer = CrossAttentionLlamaModel(config, 
                                                            embed_wsi=True, 
                                                            wsi_num_hidden_layers=1)

        num_layers = len(self.model.layers)
        print(config)
        self.cross_attn = config.cross_attn
        cross_attn_layer_index = config.cross_attn_layer_index
        # Choose which layers to modify for cross attention
        self.model.layers[cross_attn_layer_index] = CrossAttentionLlamaDecoderLayer(config, 
                                                                                    layer_idx=cross_attn_layer_index)

        self.pretraining_tp = getattr(config, "pretraining_tp", False)
        self.training = getattr(config, "training")
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Load weights conditionally *after* architecture is built
        checkpoint_path_str = getattr(config, 'load_checkpoint_path', None)

        if checkpoint_path_str:
            self.load_text_model(checkpoint_path_str)
        else:
            logger.info("No 'load_checkpoint_path' provided in config for a pretrained Text LLM. Initializing model without loading checkpoint weights.")
        self.post_init()

    def load_text_model(self, checkpoint_path_str: str) -> None:
        
        logger.info(f"Attempting to load weights from TextLlama checkpoint: {checkpoint_path_str}")
        checkpoint_path = Path(checkpoint_path_str)
        
        # Determine weight file path (prefer safetensors)
        safetensors_path = checkpoint_path / "model.safetensors"
        loaded_state_dict = load_safetensors(str(safetensors_path), device="cpu")
                
        # Load into the current model (self)
        # Using strict=True because architectures are assumed identical
        missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=True) 
                
        if missing_keys:
                logger.warning(f"Weights loaded, but some keys were missing in checkpoint: {missing_keys}")
        if unexpected_keys:
                logger.warning(f"Weights loaded, but some extra keys were found in checkpoint: {unexpected_keys}")
                
        if not missing_keys and not unexpected_keys:
                logger.info(f"Successfully loaded all weights from checkpoint: {checkpoint_path}")
        else:
                logger.warning(f"Issues encountered loading weights from {checkpoint_path}. Check warnings above.")
        
    def get_model(self) -> CrossAttentionLlamaModel:
        return self.model

    def prepare_wsi_for_cross_attention(self,
                                        wsi_embeddings: List[Tensor],
                                        padding_side: str,
                                        device: str) -> Tensor:
        
        # Padding to the max sequence length
        # This manual padding is only necessary when the input lengths differ in the batch
        # else only the attention mask will be padded and the new labels.
        batch_size = len(wsi_embeddings)
        max_len = max(x.shape[0] for x in wsi_embeddings)
        new_embeds_padded = []

        for i, wsi_embedding in enumerate(wsi_embeddings):
            cur_len = wsi_embedding.shape[0]
            padding_needed = max_len - cur_len

            # padding is the eot token
            # TODO: do this cleanly
            pad_tensor = torch.ones((padding_needed,), 
                                    dtype=torch.int, 
                                    device=device)
            
            pad_embeds = self.model.embed_tokens(pad_tensor)

            if padding_side == "left":
                new_embeds_padded.append(torch.cat([pad_embeds, wsi_embedding], dim=0))
            else:
                new_embeds_padded.append(torch.cat([wsi_embedding, pad_embeds], dim=0))

        # we'll use "<|reserved_special_token_0|>" which is token number 2
        # to represent a WSI

        wsi_embed_token = torch.ones((batch_size,1,wsi_embeddings[0].shape[1]), device=device)

        new_input_embeds = torch.stack(new_embeds_padded, dim=0)
        new_input_embeds = torch.cat([new_input_embeds, wsi_embed_token], dim=1)
        return new_input_embeds


    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids: Tensor, 
        position_ids: Tensor, 
        attention_mask: Tensor, past_key_values,
        labels: Tensor, 
        wsi_embeddings: list, 
        tokenizer_model_max_length: int = None,
        padding_side: str = "right"
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        model = self.get_model()

        new_input_embeds_list = []
        new_labels_list = []

        for i in range(batch_size):
            text_ids = input_ids[i]
            
            txt_embeds = model.embed_tokens(text_ids)
            wsi_embeds = wsi_embeddings[i]
            
            # new_embeds looks like: [wsi_embeds, bos, txt_embeds, eos]
            # bos and eos should be already in the input_ids (txt_embeds here)
            new_embeds = torch.cat([wsi_embeds, txt_embeds], dim=0)
            new_input_embeds_list.append(new_embeds)
            
            num_wsi = wsi_embeds.size(0)
            text_labels = text_ids.clone() if labels is None else labels[i]
            labels_i = torch.cat([
                torch.full((num_wsi,), IGNORE_INDEX, device=device, dtype=text_labels.dtype),
                text_labels,  # No need to Shift labels for causal LM, huggingface does this internally
            ], dim=0)
            new_labels_list.append(labels_i)
        
        # Apply truncation if necessary
        if tokenizer_model_max_length is not None:
            new_input_embeds_list = [x[:tokenizer_model_max_length] for x in new_input_embeds_list]
            new_labels_list = [x[:tokenizer_model_max_length] for x in new_labels_list]

        # Padding to the max sequence length
        # This manual padding is only necessary when the input lengths differ in the batch
        # else only the attention mask will be padded and the new labels.
        # TODO: these are using zero vectors as padding, need the padding embed
        max_len = max(x.shape[0] for x in new_input_embeds_list)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels_list[0].dtype, device=device)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        position_ids_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds_list, new_labels_list)):
            cur_len = cur_new_embed.shape[0]
            padding_needed = max_len - cur_len
            pad_tensor = torch.zeros((padding_needed, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=device)

            if padding_side == "left":
                new_input_embeds_padded.append(torch.cat([pad_tensor, cur_new_embed], dim=0))
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask_padded[i, -cur_len:] = True
                position_ids_padded[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=device)
            else:
                new_input_embeds_padded.append(torch.cat([cur_new_embed, pad_tensor], dim=0))
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask_padded[i, :cur_len] = True
                position_ids_padded[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # position_ids ommitted for now
        return None, None, attention_mask_padded, past_key_values, new_input_embeds, new_labels_padded

    def cross_attn_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        wsi_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        if self.cross_attn:
            wsi_embeddings = self.prepare_wsi_for_cross_attention(wsi_embeds, 
                                                                  padding_side="left", 
                                                                  device=input_ids.device)
            
            # take the embedding from the last token which represents the wsi sequence
            wsi_embeddings = self.wsi_embedding_layer(inputs_embeds=wsi_embeddings)[0][:,-1,:]
            wsi_embeddings = wsi_embeddings.unsqueeze(1)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            wsi_embeds=wsi_embeddings,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None,
                inputs_embeds=None, 
                labels=None, 
                use_cache=None, 
                output_attentions=None,
                output_hidden_states=None, 
                wsi_embeddings=None, 
                return_dict=None, 
                cache_position=None, 
                **kwargs):            

        # Generation step (after first token) — skip multimodal prep
        if past_key_values is not None and not self.cross_attn:
            if inputs_embeds is None:
                inputs_embeds = self.get_model().embed_tokens(input_ids)
            # Avoid conflict: must specify exactly one of input_ids or inputs_embeds
            input_ids = None

        # inputs_embeds is already available during autoregression
        # During a training forward pass inputs_embeds is None.
        # So we have to construct it.
        elif inputs_embeds is None and not self.cross_attn:
                (
                    input_ids, position_ids, attention_mask, past_key_values,
                    inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values,
                    labels, wsi_embeddings,
                    tokenizer_model_max_length=getattr(self.config, "tokenizer_model_max_length", None),
                    padding_side="right"
                )

        return self.cross_attn_forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            wsi_embeds=wsi_embeddings,
            use_cache=use_cache, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
            **kwargs
        )

    @torch.no_grad()
    def generate(self, 
                 inputs=None, 
                 wsi_embeddings=None, 
                 **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if wsi_embeddings is not None and not self.cross_attn:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, 
                position_ids, 
                attention_mask, 
                None, 
                None, 
                wsi_embeddings,
                tokenizer_model_max_length=getattr(self.config, "tokenizer_model_max_length", None),
                padding_side="right"
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if self.cross_attn:
            max_new_tokens = kwargs.pop("max_new_tokens", None)
            eos_token_id = kwargs.pop("eos_token_id", None)
            temperature = kwargs.pop("temperature", None)

            return self.custom_generate(input_ids=inputs,
                                        attention_mask=attention_mask,
                                        wsi_embeddings=wsi_embeddings,
                                        use_cache=True,
                                        temperature=temperature,
                                        max_new_tokens=max_new_tokens,
                                        eos_token_id=eos_token_id)

        return super().generate(position_ids=position_ids, 
                                attention_mask=attention_mask,
                                use_cache=True,
                                inputs_embeds=inputs_embeds, 
                                input_ids=inputs, 
                                **kwargs)
    
    @torch.no_grad()
    def custom_generate(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                wsi_embeddings: torch.Tensor,
                use_cache: bool = False,
                temperature: float = 1.0,
                max_new_tokens: int = 50,
                eos_token_id: int = None):
        """
        Custom generate() without past_key_values. Recomputes full sequence each step.
        
        Args:
            input_ids: Tensor of shape [B, T_text]
            wsi_embeddings: Tensor of shape [B, N_wsi, D]
            tokenizer: Tokenizer to embed input_ids
            temperature: Sampling temperature
            max_new_tokens: Max new tokens to generate
            eos_token_id: Token ID to stop generation (e.g. tokenizer.eos_token_id)

        Returns:
            Tensor of shape [B, T_text + max_new_tokens]
        """

        generated_ids = input_ids.clone()
        kv_cache = None

        for step in range(max_new_tokens):
            
            outputs = self.forward(
                input_ids=generated_ids[:,-1].unsqueeze(0),
                attention_mask=attention_mask,
                wsi_embeddings=wsi_embeddings,
                use_cache=use_cache,
                past_key_values=kv_cache,
                return_dict=True
            )

            logits = outputs.logits[:, -1, :]  # take last token
            next_token = torch.softmax(logits / temperature, dim=-1).multinomial(num_samples=1)  # [B, 1]

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # keep track of past_key_values
            kv_cache = outputs.past_key_values

            new_attention_mask = torch.ones((attention_mask.size(0), 1), 
                                        dtype=attention_mask.dtype, 
                                        device=attention_mask.device)
        
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

            # Stop if EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated_ids