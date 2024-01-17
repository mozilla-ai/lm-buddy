import pytest
from pydantic import ValidationError

from flamingo.integrations.wandb import WandbRunConfig


@pytest.fixture
def wandb_run_config():
    return WandbRunConfig(
        name="run-name",
        run_id="run-id",
        project="cortex-research",
        entity="twitter.com",
    )


def test_serde_round_trip(wandb_run_config):
    assert WandbRunConfig.parse_raw(wandb_run_config.json()) == wandb_run_config


def test_wandb_path(wandb_run_config):
    assert wandb_run_config.wandb_path == "twitter.com/cortex-research/run-id"


def test_ensure_run_id():
    env = WandbRunConfig(name="defined", project="defined", entity="defined")
    assert env.run_id is not None  # Pydantic validator fills this in


def test_env_vars(wandb_run_config):
    env_vars = wandb_run_config.get_env_vars()
    expected = ["WANDB_NAME", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID"]
    for key in expected:
        assert key in env_vars
    assert "WANDB_RUN_GROUP" not in env_vars


def test_disallowed_kwargs():
    with pytest.raises(ValidationError):
        WandbRunConfig(name="name", project="project", old_name="I will throw")


def test_missing_key_warning(mock_environment_without_keys):
    with pytest.warns(UserWarning):
        config = WandbRunConfig(name="I am missing an API key", project="I should warn the user")
        assert "WANDB_API_KEY" not in config.get_env_vars()
