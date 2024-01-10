from pathlib import Path

import pytest
from flamingo.jobs.configs import LMHarnessJobConfig
from pydantic import ValidationError


def test_bad_hf_name(default_lm_harness_config):
    with pytest.raises(ValidationError):
        default_lm_harness_config(
            model_name_or_path="dfa../invalid",
        )


def test_serde_round_trip_path_notmp(mock_wandb_env, default_lm_harness_config):
    config = default_lm_harness_config(
        model_name_or_path=Path("test"),
        wandb_env=None,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_checkpointpath_only(
    mock_wandb_env, checkpoint_path, default_lm_harness_config
):
    config = default_lm_harness_config(
        model_name_or_path=checkpoint_path,
        wandb_env=None,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_wandb_and_path(
    mock_wandb_env, checkpoint_path, default_lm_harness_config
):
    config = default_lm_harness_config(
        model_name_or_path=checkpoint_path,
        wandb_env=mock_wandb_env(),
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_wandb(mock_wandb_env, default_lm_harness_config):
    config = default_lm_harness_config(
        model_name_or_path=None,
        wandb_env=mock_wandb_env(),
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_with_checkpoint_path_no_wandb(default_lm_harness_config):
    config = default_lm_harness_config(
        model_name_or_path="/fake_path/to/a/file",
        wandb_env=None,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_serde_round_trip_no_wandb(default_lm_harness_config):
    config = default_lm_harness_config(
        model_name_or_path="some/model",
        wandb_env=None,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_model_validation():
    with pytest.raises(ValidationError):
        # Neither checkpoint_path or wandb_env specified
        LMHarnessJobConfig(tasks=["task1", "task2"], num_fewshot=5, batch_size=16)


def test_serde_round_trip_default_config(default_lm_harness_config):
    config = default_lm_harness_config(
        model_name_or_path="fake_path",
        wandb_env=None,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_parse_from_yaml(default_lm_harness_config, tmp_path_factory):
    config = default_lm_harness_config(model_name_or_path="not_a_real_model")
    p = tmp_path_factory.mktemp("test_yaml") / "eval.yaml"
    config.to_yaml_file(p)
    assert config == LMHarnessJobConfig.from_yaml_file(p)
