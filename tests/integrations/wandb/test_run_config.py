import pytest
from pydantic import ValidationError

from flamingo.integrations.wandb import WandbRunConfig


def test_env_vars(wandb_run_config_generator):
    env_vars = wandb_run_config_generator().get_env_vars()
    expected = ["WANDB_NAME", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID"]
    for key in expected:
        assert key in env_vars
    assert "WANDB_RUN_GROUP" not in env_vars


def test_serde_round_trip(wandb_run_config_generator):
    assert (
        WandbRunConfig.parse_raw(wandb_run_config_generator().json())
        == wandb_run_config_generator()
    )


def test_disallowed_kwargs():
    with pytest.raises(ValidationError):
        WandbRunConfig(name="name", project="project", old_name="I will throw")


def test_missing_key_warning(mock_environment_without_keys):
    with pytest.warns(UserWarning):
        env = WandbRunConfig(name="I am missing an API key", project="I should warn the user")
        assert "WANDB_API_KEY" not in env.env_vars
