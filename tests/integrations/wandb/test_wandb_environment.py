import pytest
from flamingo.integrations.wandb import WandbEnvironment
from pydantic import ValidationError


def test_env_vars(default_wandb_env):
    env_vars = default_wandb_env().env_vars
    expected = ["WANDB_NAME", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID"]
    for key in expected:
        assert key in env_vars
    assert "WANDB_RUN_GROUP" not in env_vars


def test_serde_round_trip(default_wandb_env):
    assert WandbEnvironment.parse_raw(default_wandb_env().json()) == default_wandb_env()


def test_disallowed_kwargs():
    with pytest.raises(ValidationError):
        WandbEnvironment(name="name", project="project", old_name="I will throw")


def test_missing_key_warning(mock_environment_without_keys):
    with pytest.warns(UserWarning):
        env = WandbEnvironment(name="I am missing an API key", project="I should warn the user")
        assert "WANDB_API_KEY" not in env.env_vars
