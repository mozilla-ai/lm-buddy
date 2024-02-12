import pytest
from pydantic import ValidationError

from lm_buddy.integrations.huggingface import DatasetConfig, HuggingFaceRepoConfig


def test_split_is_required():
    with pytest.raises(ValidationError):
        repo = HuggingFaceRepoConfig(repo_id="dataset/xyz")
        DatasetConfig(load_from=repo, split=None)
