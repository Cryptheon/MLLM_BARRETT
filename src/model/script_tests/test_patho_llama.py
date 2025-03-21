import torch
from patho_llama import PathoLlamaForCausalLM, PathoLlamaConfig

class DummyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def add_tokens(self, tokens, special_tokens=False):
        count = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                count += 1
        return count

    def __len__(self):
        return len(self.vocab)

def initialize_model():
    """Initialize the model, config, and tokenizer."""
    config = PathoLlamaConfig(
        vocab_size=32000, hidden_size=768, num_hidden_layers=4,
        num_attention_heads=4, max_position_embeddings=512, pad_token_id=0,
        bos_token_id=1, eos_token_id=2
    )
    
    model = PathoLlamaForCausalLM(config)
    tokenizer = DummyTokenizer()
    tokenizer.add_tokens(["<eoi>", "<eos>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    return config, model, tokenizer

def test_forward(model):
    """Test the forward pass."""
    batch_size, seq_len = 1, 50
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    wsi_embeddings = [
        torch.randn(2, model.config.hidden_size),
        torch.randn(3, model.config.hidden_size)
    ]

    outputs = model(input_ids=input_ids, wsi_embeddings=wsi_embeddings)
    logits = outputs.logits
    print("Forward pass logits shape:", logits.shape)
    print("loss", outputs.loss)

def test_generate(model):
    """Test the generate method."""
    model.training = False
    batch_size, seq_len = 1, 50
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    
    wsi_embeddings = [
        torch.randn(2, model.config.hidden_size),
        torch.randn(3, model.config.hidden_size),
    ]
    
    generated_ids = model.generate(inputs=input_ids, wsi_embeddings=wsi_embeddings, max_new_tokens=20)
    print("Generated token IDs shape:", generated_ids.shape)
    print("Generated token IDs:", generated_ids)

def main():
    """Main function to run tests."""
    _, model, _ = initialize_model()
    print("=== Testing Forward Pass ===")
    test_forward(model)
    print("\n=== Testing Generate Method ===")
    test_generate(model)

if __name__ == "__main__":
    main()
