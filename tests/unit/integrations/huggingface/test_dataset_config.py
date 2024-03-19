import pytest
from pydantic import ValidationError

from lm_buddy.integrations.huggingface import DatasetConfig


def test_split_is_required():
    with pytest.raises(ValidationError):
        DatasetConfig(path="hf://dataset/xyz", split=None)
