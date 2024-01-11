from pathlib import Path

import pytest
from flamingo.jobs.configs import LMHarnessJobConfig
from pydantic import ValidationError


def test_bad_hf_name(default_lm_harness_config):
    with pytest.raises(ValidationError):
        default_lm_harness_config(model_name_or_path="dfa../invalid")


def test_serde_round_trip_default_config(default_lm_harness_config):
    config = default_lm_harness_config()
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_with_path(default_lm_harness_config):
    config = default_lm_harness_config(model_name_or_path=Path("fake/path"))
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_parse_from_yaml(default_lm_harness_config, tmp_path_factory):
    config = default_lm_harness_config(model_name_or_path="not_a_real_model")
    p = tmp_path_factory.mktemp("test_yaml") / "eval.yaml"
    config.to_yaml_file(p)
    assert config == LMHarnessJobConfig.from_yaml_file(p)
