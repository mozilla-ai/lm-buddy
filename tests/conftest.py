"""
Tests for the LLM tuner. This file is used to provide fixtures for the test session 
that are accessible to all submodules.
"""
import dataclasses
import os
from collections.abc import Generator
from unittest import mock

import pytest
import tuner
import wandb
from tuner.integrations.wandb import WandbEnvironment
from tuner.jobs.evaluation_config import EvaluationJobConfig
from wandb.sdk.lib.runid import generate_id


@dataclasses.dataclass
class Project:
    name: str


@pytest.fixture(autouse=True, scope="function")
def mock_wandb_api_call():
    with mock.patch(
        "tuner.integrations.wandb.utils.wandb.Api.projects", return_value=[Project("my-project")]
    ) as p:
        yield p


### from wandb
def dict_factory():
    def helper():
        return dict()

    return helper


@pytest.fixture(scope="function")
def test_settings():
    """
    taken from wandb to generate a test Settings instance.
    """

    def update_test_settings(
        extra_settings: dict | wandb.sdk.wandb_settings.Settings = dict_factory(),  # noqa: B008
    ):
        settings = wandb.Settings(
            console="off",
            save_code=False,
        )
        if isinstance(extra_settings, dict):
            settings.update(extra_settings, source=wandb.sdk.wandb_settings.Source.BASE)
        elif isinstance(extra_settings, wandb.sdk.wandb_settings.Settings):
            settings.update(extra_settings)
        settings._set_run_start_time()
        return settings

    yield update_test_settings


@pytest.fixture(scope="function")
def mock_run(test_settings):
    """
    taken from wandb to generate a test Run instance.
    """
    from wandb.sdk.lib.module import unset_globals

    def mock_run_fn(**kwargs) -> "wandb.sdk.wandb_run.Run":
        kwargs_settings = kwargs.pop("settings", dict())
        kwargs_settings = {
            **{
                "run_id": generate_id(),
            },
            **kwargs_settings,
        }
        run = wandb.wandb_sdk.wandb_run.Run(settings=test_settings(kwargs_settings), **kwargs)
        run._set_backend(mock.MagicMock())
        run._set_globals()
        return run

    yield mock_run_fn
    unset_globals()


@pytest.fixture(autouse=True, scope="function")
def mock_valid_run(mock_run):
    """
    taken from wandb to generate a valid run.
    """
    run = mock_run()
    with mock.patch("tuner.integrations.wandb.utils._resolve_wandb_run", return_value=run) as r:
        yield r


@pytest.fixture(autouse=True, scope="function")
def mock_environment_with_keys():
    """Mocks an API key-like mechanism for the environment."""
    with mock.patch.dict(os.environ, {"WANDB_API_KEY": "abcdefg123"}):
        yield


@pytest.fixture(autouse=True, scope="function")
def mock_environment_without_keys():
    """Mocks an environment missing common API keys."""
    with mock.patch.dict(os.environ, clear=True):
        yield


@pytest.fixture(scope="function")
def mock_wandb_env(mock_run, test_settings) -> Generator[WandbEnvironment, None, None]:
    """
    Sets up a mock wandb_env object.
    """

    def mock_env_func(**kwargs) -> "tuner.integrations.wandb.WandbEnvironment":
        mine = {
            "name": "my-run",
            "project": "my-project",
            "entity": "mozilla-ai",
            "run_id": "gabbagool-123",
        }
        kwargs = {**mine, **kwargs}
        return WandbEnvironment(**kwargs)

    yield mock_env_func


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory):
    "makes a mocked checkpoint path dir."
    fn = tmp_path_factory.mktemp("data") / "model"
    return fn


@pytest.fixture(scope="function")
def default_eval_config(mock_run, test_settings):
    """
    Sets up a default
    """

    def default_eval_config(**kwargs) -> "tuner.integrations.wandb.WandbEnvironment":
        mine = {
            "tasks": ["task1", "task2"],
            "model_name_or_path": None,
            "wandb_env": None,
            "num_fewshot": 5,
            "batch_size": 16,
            "torch_dtype": "bfloat16",
            "quantization_config": None,
            "timeout": 3600,
        }
        return EvaluationJobConfig(**{**mine, **kwargs})

    yield default_eval_config
