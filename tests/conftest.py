"""
Tests for the Flamingo. 

This file is used to provide fixtures for the test session that are accessible to all submodules.
"""
import os
from unittest import mock

import pytest

from flamingo.integrations.wandb import WandbArtifactConfig, WandbRunConfig
from flamingo.jobs import LMHarnessJobConfig


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
def default_wandb_run_config():
    def generator(**kwargs) -> WandbRunConfig:
        mine = {
            "name": "my-run",
            "project": "my-project",
            "entity": "mozilla-ai",
            "run_id": "gabbagool-123",
        }
        return WandbRunConfig(**{**mine, **kwargs})

    yield generator


@pytest.fixture(scope="function")
def default_wandb_artifact_config():
    def generator(**kwargs) -> WandbArtifactConfig:
        mine = {
            "name": "my-run",
            "version": "latest",
            "project": "research-project",
            "entity": "mozilla-corporation",
        }
        return WandbArtifactConfig(**{**mine, **kwargs})

    yield generator


@pytest.fixture(scope="function")
def default_lm_harness_config():
    def generator(**kwargs) -> LMHarnessJobConfig:
        mine = {
            "tasks": ["task1", "task2"],
            "num_fewshot": 5,
            "batch_size": 16,
            "torch_dtype": "bfloat16",
            "model_name_or_path": None,
            "quantization": None,
            "timeout": 3600,
        }
        return LMHarnessJobConfig(**{**mine, **kwargs})

    yield generator
