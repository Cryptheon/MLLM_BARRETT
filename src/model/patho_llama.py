import logging
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch.nn as nn

IGNORE_INDEX = -100
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PathoLlamaConfig(LlamaConfig):
    model_type = "patho_llama"
    training = True

class PathoLlamaModel(LlamaModel):
    config_class = PathoLlamaConfig
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        logger.debug(f"Initialized PathoLlamaModel with config: {config}")

class PathoLlamaForCausalLM(LlamaForCausalLM):
    config_class = PathoLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = PathoLlamaModel(config)
        self.pretraining_tp = getattr(config, "pretraining_tp", False)
        self.training = getattr(config, "training")
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Load weights conditionally *after* architecture is built
        checkpoint_path_str = getattr(config, 'load_checkpoint_path', None)

        if checkpoint_path_str:
            self.load_text_model(checkpoint_path_str)
        else:
            logger.info("No pretrained base model from 'load_checkpoint_path' provided in config. Initializing model without loading pretrained checkpoint weights.")
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
        
    def get_model(self) -> PathoLlamaModel:
        return self.model

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
        # This manual padding is only necessary when the input lengths differ
        # else only the attention mask will be padded and the new labels.
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
        if past_key_values is not None:
            if inputs_embeds is None:
                inputs_embeds = self.get_model().embed_tokens(input_ids)
            # Avoid conflict: must specify exactly one of input_ids or inputs_embeds
            input_ids = None
            # for generation we don't need to compute against the labels  
            labels = None
            
        # inputs_embeds is already available during autoregression
        # During a training forward pass inputs_embeds is None.
        # So we have to construct it.
        else:
            if inputs_embeds is None:
                (
                    input_ids, position_ids, attention_mask, past_key_values,
                    inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values,
                    labels, wsi_embeddings,
                    tokenizer_model_max_length=getattr(self.config, "tokenizer_model_max_length", None),
                    padding_side="right"
                )

        return super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels,
            use_cache=use_cache, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
            cache_position=cache_position, 
            **kwargs
        )

    @torch.no_grad()
    def generate(self, 
                 inputs=None, 
                 wsi_embeddings=None, 
                 **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if wsi_embeddings is not None:
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


        return super().generate(position_ids=position_ids, 
                                attention_mask=attention_mask,
                                use_cache=True,
                                inputs_embeds=inputs_embeds, 
                                input_ids=None, #inputs
                                **kwargs)
    
    # @torch.no_grad()
    # def custom_generate(self,
    #             input_ids: torch.Tensor,
    #             wsi_embeddings: torch.Tensor,
    #             temperature: float = 1.0,
    #             max_new_tokens: int = 50,
    #             eos_token_id: int = None):
    #     """
    #     Custom generate() without past_key_values. Recomputes full sequence each step.
        
    #     Args:
    #         input_ids: Tensor of shape [B, T_text]
    #         wsi_embeddings: Tensor of shape [B, N_wsi, D]
    #         tokenizer: Tokenizer to embed input_ids
    #         temperature: Sampling temperature
    #         max_new_tokens: Max new tokens to generate
    #         eos_token_id: Token ID to stop generation (e.g. tokenizer.eos_token_id)

    #     Returns:
    #         Tensor of shape [B, T_text + max_new_tokens]
    #     """
    #     device = input_ids.device
    #     batch_size = input_ids.size(0)

    #     generated_ids = input_ids.clone()

    #     for step in range(max_new_tokens):
    #         # Rebuild full embeddings: [WSI, input_ids, gen_1, ..., gen_n]
    #         text_embeds = self.get_model().embed_tokens(generated_ids)
    #         full_embeds = torch.cat([wsi_embeddings, text_embeds], dim=1)  # [B, total_len, D]
    #         attention_mask = torch.ones(full_embeds.size()[:2], dtype=torch.bool, device=device)

    #         outputs = super().forward(
    #             inputs_embeds=full_embeds,
    #             attention_mask=attention_mask,
    #             use_cache=False,
    #             return_dict=True
    #         )

    #         logits = outputs.logits[:, -1, :]  # take last token
    #         next_token = torch.softmax(logits / temperature, dim=-1).multinomial(num_samples=1)  # [B, 1]

    #         # Append token
    #         generated_ids = torch.cat([generated_ids, next_token], dim=1)

    #         # Stop if EOS
    #         if eos_token_id is not None and (next_token == eos_token_id).all():
    #             break

    #     return generated_ids