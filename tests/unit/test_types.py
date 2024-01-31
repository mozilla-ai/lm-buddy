import pytest
import torch
from pydantic import ValidationError

from flamingo.jobs.simple import SimpleJobConfig
from flamingo.types import BaseFlamingoConfig, TorchDtypeString


def test_base_config_settings():
    class DummyConfig(BaseFlamingoConfig):
        value: int

    # Validate assignment
    config = DummyConfig(value=42)
    with pytest.raises(ValidationError):
        config.value = "dogs"

    # Extra forbid
    with pytest.raises(ValidationError):
        DummyConfig(value=42, foo="bar")


def test_torch_dtype_validation():
    # Valid dtype strings
    x = TorchDtypeString.validate("int64")
    assert x == "int64"
    assert x.as_torch() == torch.int64
    x = TorchDtypeString.validate(torch.int64)
    assert x == "int64"
    assert x.as_torch() == torch.int64
    # Invalid dtype strings
    with pytest.raises(ValueError):
        TorchDtypeString.validate(5)
    with pytest.raises(ValueError):
        TorchDtypeString.validate("dogs")


def test_config_as_tempfile():
    config = SimpleJobConfig(magic_number=42)
    config_name = "my-special-config.yaml"
    with config.to_tempfile(name=config_name) as path:
        assert path.name == config_name
        assert SimpleJobConfig.from_yaml_file(path) == config
