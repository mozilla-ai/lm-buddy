from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # Base config for an already tiny model
    base_model = "prajjwal1/bert-tiny"
    config = AutoConfig.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Make it even smaller
    config.hidden_size = 4
    config.intermediate_size = 4
    config.num_hidden_layers = 1
    config.num_attention_heads = 1
    config._name_or_path = "bert-tiniest"

    # Initialize a model with totally random weights (not actually "pretrained")
    model = AutoModelForCausalLM.from_config(config)

    # Save as repo resources
    save_path = Path(__file__).parent
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
