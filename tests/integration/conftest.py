"""
Flamingo integration test suite.

These tests generally require a running Ray cluster/W&B environment.
"""
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import ray


@pytest.fixture(autouse=True, scope="module")
def temporary_storage_path():
    with tempfile.TemporaryDirectory() as storage_path:
        yield Path(storage_path)


@pytest.fixture(autouse=True, scope="module")
def integration_environment(temporary_storage_path):
    """Environment variables for running integration tests.

    Includes the following:
    - Setting the storage path for Ray Train
    - Disabling the W&B SDK and re-routing cache directories
    - Adding fake API keys for W&B, OpenAI, etc..
    """
    it_env = {
        "RAY_STORAGE": str(temporary_storage_path / "ray_results"),
        "WANDB_DIR": str(temporary_storage_path / "wandb" / "logs"),
        "WANDB_CACHE_DIR": str(temporary_storage_path / "wandb" / "cache"),
        "WANDB_CONFIG_DIR": str(temporary_storage_path / "wandb" / "configs"),
        "WANDB_MODE": "disabled",
        "WANDB_API_KEY": "SECRET",
        "OPENAI_API_KEY": "SECRET",
    }
    with mock.patch.dict(os.environ, it_env):
        yield it_env


@pytest.fixture(autouse=True, scope="module")
def initialize_ray_cluster(integration_environment):
    """Initialize a small, fixed-size Ray cluster for testing.

    Ray has other options for testing that we could explore down the road
    (https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html).
    But for now, a small, static-size cluster as a fixture seems to work fine.
    """
    try:
        # Num CPUs is auto-detected so we don't need to pass it
        ray.init(
            runtime_env={"env_vars": integration_environment},
            num_gpus=0,
        )
        yield
    finally:
        ray.shutdown()
