import torch
import logging
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from typing import Optional, Tuple, List, Union 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextLlamaConfig(LlamaConfig):
    """
    Configuration class for a text-only Llama model, inheriting from LlamaConfig.
    """
    model_type = "text_llama" 

class TextLlamaModel(LlamaModel):
    """
    Base Llama model wrapper for text-only pretraining.
    Currently identical to LlamaModel but uses TextLlamaConfig.
    """
    config_class = TextLlamaConfig
    def __init__(self, config: TextLlamaConfig):
        super().__init__(config)
        logger.debug(f"Initialized TextLlamaModel with config: {config}")

class TextLlamaForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM adapted for text-only pretraining.
    Removes multimodal handling (WSI embeddings).
    """
    config_class = TextLlamaConfig # Use the text-specific config

    def __init__(self, config: TextLlamaConfig):
        """
        Initializes the TextLlamaForCausalLM model.

        Args:
            config (TextLlamaConfig): Model configuration.
        """
        super().__init__(config) # Initialize the base LlamaForCausalLM

        self.model = TextLlamaModel(config) 
        
        # Keep standard LM head initialization from LlamaForCausalLM
        # Verify if LlamaForCausalLM's __init__ handles vocab_size and lm_head setup.
        # If it does, these lines might be redundant unless customizing the head.
        # self.vocab_size = config.vocab_size 
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        
        logger.info("Initialized TextLlamaForCausalLM for text-only pretraining.")

    def get_model(self) -> TextLlamaModel:
        """Returns the base TextLlamaModel instance."""
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # For newer HF versions
        # Removed wsi_embeddings parameter
        **kwargs, # Accept extra kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Standard forward pass for a text-based Causal LM.

        Relies on the parent LlamaForCausalLM's forward method after ensuring
        inputs are standard text inputs (no multimodal preparation needed).
        """

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