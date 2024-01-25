"""
Tests for the Flamingo. 

This file is used to provide fixtures for the test session that are accessible to all submodules.
"""
import os
from pathlib import Path
from unittest import mock

import pytest
import ray


@pytest.fixture
def examples_folder():
    return Path(__file__).parents[1] / "examples"


@pytest.fixture
def resources_folder():
    return Path(__file__).parent / "resources"


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


@pytest.fixture(autouse=True, scope="function")
def mock_environment_wandb_disabled():
    """Mocks an environment with W&B disabled."""
    with mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
        yield


@pytest.fixture(scope="session")
def initialize_ray_cluster():
    try:
        ray.init(
            # Auto-detect num_cpu
            num_gpus=0,
            runtime_env={"env_vars": {"WANDB_MODE": "disabled"}},
        )
        yield
    finally:
        ray.shutdown()
