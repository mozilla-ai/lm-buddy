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
        yield storage_path


@pytest.fixture(autouse=True, scope="module")
def disabled_wandb_env(temporary_storage_path):
    """Inject environment variables to disable W&B logging.

    This should turn calls to the W&B SDK into no-ops
    (https://docs.wandb.ai/guides/technical-faq/general#can-i-disable-wandb-when-testing-my-code),
    but may still write some local files, hence the tmp directory injection.

    We may want to re-evalaute this and instead set W&B to `offline` mode
    down the line because it maintains SDK functionality but simply stops cloud syncing.
    """
    storage = Path(temporary_storage_path)
    wandb_env = {
        "WANDB_DIR": str(storage / "wandb" / "logs"),
        "WANDB_CACHE_DIR": str(storage / "wandb" / "cache"),
        "WANDB_CONFIG_DIR": str(storage / "wandb" / "configs"),
        "WANDB_API_KEY": "MY-API-KEY",
        "WANDB_MODE": "disabled",
    }
    with mock.patch.dict(os.environ, wandb_env):
        yield wandb_env


@pytest.fixture(autouse=True, scope="module")
def initialize_ray_cluster(disabled_wandb_env):
    """Initialize a small, fixed-size Ray cluster for testing.

    Ray has other options for testing that we could explore down the road
    (https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html).
    But for now, a small, static-size cluster as a fixture seems to work fine.
    """
    try:
        # Not passing -> auto-detect num_cpu
        ray.init(
            runtime_env={"env_vars": disabled_wandb_env},
            num_gpus=0,
        )
        yield
    finally:
        ray.shutdown()
