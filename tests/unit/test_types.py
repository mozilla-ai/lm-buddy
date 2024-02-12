import pytest
import torch
from pydantic import ValidationError

from lm_buddy.jobs.simple import SimpleJobConfig
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


def test_config_as_tempfile():
    config = SimpleJobConfig(magic_number=42)
    config_name = "my-special-config.yaml"
    with config.to_tempfile(name=config_name) as path:
        assert path.name == config_name
        assert SimpleJobConfig.from_yaml_file(path) == config
