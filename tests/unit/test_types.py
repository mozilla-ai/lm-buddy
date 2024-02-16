import pytest
import torch
from pydantic import ValidationError

from lm_buddy.types import BaseLMBuddyConfig, SerializableTorchDtype


def test_base_config_settings():
    class TestConfig(BaseLMBuddyConfig):
        value: int

    # Validate assignment
    config = TestConfig(value=42)
    with pytest.raises(ValidationError):
        config.value = "dogs"

    # Extra forbid
    with pytest.raises(ValidationError):
        TestConfig(value=42, foo="bar")


def test_serializable_torch_dtype():
    class TestConfig(BaseLMBuddyConfig):
        torch_dtype: SerializableTorchDtype

    config = TestConfig(torch_dtype="bfloat16")
    assert config.torch_dtype == torch.bfloat16

    # Invalid dtypes
    with pytest.raises(ValueError):
        TestConfig(torch_dtype=5)
    with pytest.raises(ValueError):
        TestConfig(torch_dtype="dogs")
