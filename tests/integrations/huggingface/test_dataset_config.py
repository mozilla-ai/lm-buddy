import pytest
from pydantic import ValidationError

from flamingo.integrations.huggingface import DatasetConfig, HuggingFaceRepoConfig


def test_split_is_required():
    with pytest.raises(ValidationError):
        repo = HuggingFaceRepoConfig(repo_id="dataset/xyz")
        DatasetConfig(load_from=repo, split=None)
