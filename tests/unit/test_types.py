import pytest
import torch

from flamingo.jobs.simple import SimpleJobConfig
from flamingo.types import TorchDtypeString


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
