import pytest

import lm_buddy
from lm_buddy.integrations.huggingface import AutoModelConfig


def test_invalid_config_error():
    not_a_job_config = AutoModelConfig(load_from="distilgpt2")
    with pytest.raises(ValueError):
        lm_buddy.run_job(not_a_job_config)
