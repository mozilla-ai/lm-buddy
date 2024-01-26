import pytest
import torch

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
