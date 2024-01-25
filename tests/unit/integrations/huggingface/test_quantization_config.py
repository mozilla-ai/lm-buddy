import pytest
import torch

from flamingo.integrations.huggingface import QuantizationConfig


@pytest.fixture
def quantization_config() -> QuantizationConfig:
    return QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def test_serde_round_trip(quantization_config: QuantizationConfig):
    assert QuantizationConfig.parse_raw(quantization_config.json()) == quantization_config
