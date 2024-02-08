"""Script to generate the `fake_gpt2` model and tokenizer files."""
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # Load base HF config and tokenizer
    hf_model_name = "openai-community/gpt2"
    config = AutoConfig.from_pretrained(hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Make the config super tiny
    config._name_or_path = "fake-gpt2"
    config.n_head = 1
    config.n_layer = 1
    config.n_embd = 2

    # Instantiate a model with random weights (not actually "pretrained")
    model = AutoModelForCausalLM.from_config(config)

    # Save model and tokenizer as resources
    save_path = Path(__file__).parent
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
