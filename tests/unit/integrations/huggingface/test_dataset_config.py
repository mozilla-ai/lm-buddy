import pytest
from pydantic import ValidationError

from lm_buddy.integrations.huggingface import DatasetConfig
from lm_buddy.paths import AssetPath


def test_split_is_required():
    with pytest.raises(ValidationError):
        dataset_path = AssetPath.from_huggingface_repo("imdb")
        DatasetConfig(path=dataset_path, split=None)
