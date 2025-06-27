import json
import time
import os
from typing import List, Dict, Any

# Import both libraries, as this module now supports both backends
# Make sure to install them: pip install llama-cpp-python openai
import openai
from llama_cpp import Llama

class ModelProcessor:
    """
    A unified class to encapsulate model interaction and processing for different backends.
    It can use either a local llama-cpp model or an OpenAI-compatible API,
    selectable via the configuration file.
    """
    def __init__(self, config: Any, prompt_path: str):
        """
        Initializes the ModelProcessor based on the specified engine in the config.

        Args:
            config (Any): A configuration object. It must have an 'engine' attribute
                          set to either 'llama_cpp' or 'openai'.
            prompt_path (str): The path to the file containing the base prompt.
        """
        self.config = config
        self.base_prompt = self._get_prompt(prompt_path)
        
        # Determine which engine to use
        self.engine = getattr(self.config, 'engine', None)
        if self.engine == 'llama_cpp':
            self.model = self._initialize_llama_cpp()
        elif self.engine == 'openai':
            self.model = self._initialize_openai()
        else:
            raise ValueError("Configuration must specify an 'engine': 'llama_cpp' or 'openai'")

    def _get_prompt(self, prompt_path: str) -> str:
        """Reads the base prompt from a file."""
        with open(prompt_path, "r", encoding='utf-8') as file:
            return file.read()

    def _initialize_llama_cpp(self) -> Llama:
        """Initializes the Llama model for local processing."""
        print("Initializing Llama-cpp model...")
        try:
            model = Llama(
                model_path=self.config.model_path,
                n_gpu_layers=self.config.n_gpu_layers,
                flash_attn=True,
                n_ctx=16384 * 2,
                chat_format=self.config.chat_format,
                verbose=True
            )
            print("Llama-cpp model initialized.")
            return model
        except AttributeError as e:
            raise AttributeError(f"Llama-cpp config is missing an attribute: {e}")

    def _initialize_openai(self) -> openai.OpenAI:
        """Initializes the OpenAI client for API-based processing."""
        print("Initializing OpenAI client...")
        try:
            client = openai.OpenAI(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
            )
            print(f"OpenAI client initialized for model '{self.config.model_id}' at {self.config.api_base}")
            return client
        except AttributeError as e:
            raise AttributeError(f"OpenAI config is missing an attribute: {e}")

    def process_batch(self, json_texts: List[str], num_variations: int) -> List[List[Dict]]:
        """
        Processes a batch of JSON strings using the configured engine.
        This method dispatches to the correct backend processor.
        """
        if self.engine == 'llama_cpp':
            return self._process_batch_llama_cpp(json_texts, num_variations)
        elif self.engine == 'openai':
            return self._process_batch_openai(json_texts, num_variations)
        else:
            # This case should not be reached due to the check in __init__
            raise RuntimeError("Model engine is not properly configured.")

    def _process_batch_llama_cpp(self, json_texts: List[str], num_variations: int) -> List[List[Dict]]:
        """Backend for llama-cpp processing."""
        processed_batch_results = []
        start_time = time.time()
        # Token counting for llama-cpp can be more complex; simplified here.
        total_tokens = 0 

        for json_text in json_texts:
            single_input_variations = []
            messages = [{"role": "system", "content": "You are an expert pathology report processor outputting only valid JSON."},
                        {"role": "user", "content": self.base_prompt.replace("__JSON_TO_TRANSLATE__", json_text)}]
            
            for _ in range(num_variations):
                try:
                    output = self.model.create_chat_completion(
                        messages=messages,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        stream=False,
                        response_format={"type": "json_object"},
                    )
                    result_text = output['choices'][0]['message']['content']
                    parsed_result = json.loads(result_text)
                    single_input_variations.append(parsed_result)
                    total_tokens += output.get('usage', {}).get('total_tokens', 0)
                except Exception as e:
                    # Generic error handling for brevity
                    print(f"An error occurred with llama-cpp: {e}")
                    single_input_variations.append({"error": str(e)})

            processed_batch_results.append(single_input_variations)

        elapsed_time = time.time() - start_time
        print(f"\n[Llama-cpp] Processed {len(json_texts)} items in {elapsed_time:.2f}s.")
        return processed_batch_results

    def _process_batch_openai(self, json_texts: List[str], num_variations: int) -> List[List[Dict]]:
        """Backend for OpenAI API processing."""
        processed_batch_results = []
        start_time = time.time()
        total_tokens = 0

        for json_text in json_texts:
            single_input_variations = []
            messages = [{"role": "system", "content": "You are an expert pathology report processor outputting only valid JSON."},
                        {"role": "user", "content": self.base_prompt.replace("__JSON_TO_TRANSLATE__", json_text)}]

            for _ in range(num_variations):
                try:
                    api_params = {
                        "model": self.config.model_id, "messages": messages, "response_format": {"type": "json_object"}
                    }
                    if hasattr(self.config, 'temperature'): api_params['temperature'] = self.config.temperature
                    if hasattr(self.config, 'top_p'): api_params['top_p'] = self.config.top_p
                    if hasattr(self.config, 'max_tokens'): api_params['max_tokens'] = self.config.max_tokens

                    output = self.model.chat.completions.create(**api_params)
                    result_text = output.choices[0].message.content
                    if result_text is None: raise ValueError("API returned null content.")
                    parsed_result = json.loads(result_text)
                    single_input_variations.append(parsed_result)
                    if output.usage: total_tokens += output.usage.total_tokens
                except Exception as e:
                    print(f"An error occurred with OpenAI API: {e}")
                    single_input_variations.append({"error": str(e)})

            processed_batch_results.append(single_input_variations)
        
        elapsed_time = time.time() - start_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        print(f"\n[OpenAI] Throughput: {tokens_per_second:.2f} tokens/sec | Total: {total_tokens} tokens in {elapsed_time:.2f}s.")
        return processed_batch_results

