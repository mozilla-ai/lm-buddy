import pytest
from pydantic import ValidationError

from lm_buddy.integrations.huggingface import DatasetConfig
from lm_buddy.paths import HuggingFaceRepoID


def test_split_is_required():
    with pytest.raises(ValidationError):
        repo = HuggingFaceRepoID(repo_id="dataset/xyz")
        DatasetConfig(path=repo, split=None)
