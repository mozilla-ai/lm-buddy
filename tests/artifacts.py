import random
import string

from datasets import Dataset
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer


def generate_xyz_dataset(*, n_row: int = 100) -> Dataset:
    mappings = [{"x": i, "y": 2 * i, "z": 3 * i + 10} for i in range(n_row)]
    return Dataset.from_list(mappings)


def generate_text_dataset(
    *,
    n_row: int = 100,
    max_char_per_row: int = 16,
    text_field: str = "text",
) -> Dataset:
    alphabet = string.ascii_letters + string.digits

    def data_generator():
        for _ in range(n_row):
            n_char = random.randint(1, max_char_per_row)
            yield {text_field: "".join(random.choice(alphabet) for _ in range(n_char))}

    return Dataset.from_generator(data_generator)


def generate_gpt2_tokenizer() -> GPT2Tokenizer:
    pass


def generate_gpt2_model(
    tokenizer: GPT2Tokenizer,
    *,
    n_head: int = 1,
    n_layer: int = 1,
    n_embd: int = 2,
) -> GPT2Model:
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.bos_token_id,
        n_head=n_head,
        n_layer=n_layer,
        n_embd=n_embd,
    )
    return GPT2Model(config)
