import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import ray


@pytest.fixture(scope="module")
def temporary_storage_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def offline_wandb_storage(temporary_storage_path):
    tmpdir = Path(temporary_storage_path)
    wandb_env = {
        "WANDB_DIR": str(tmpdir / "wandb" / "logs"),
        "WANDB_CACHE_DIR": str(tmpdir / "wandb" / "cache"),
        "WANDB_CONFIG_DIR": str(tmpdir / "wandb" / "configs"),
        "WANDB_API_KEY": "MY-API-KEY",
        "WANDB_MODE": "offline",
    }
    with mock.patch.dict(os.environ, wandb_env):
        yield


@pytest.fixture(scope="module")
def initialize_ray_cluster(offline_wandb_storage):
    """Initialize a small, fixed-size Ray cluster for testing.

    Ray has other options for testing that we could explore down the road
    (https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html).
    But for now, a small, static-size cluster as a fixture seems to work fine.
    """
    try:
        # Not passing -> auto-detect num_cpu
        ray.init(
            runtime_env={"env_vars": {"WANDB_MODE": "disabled"}},
            num_gpus=0,
        )
        yield
    finally:
        ray.shutdown()
