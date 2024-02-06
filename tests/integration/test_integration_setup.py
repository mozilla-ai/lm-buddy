import os

import ray


def test_wandb_disabled():
    wandb_mode = os.environ.get("WANDB_MODE")
    assert wandb_mode == "disabled"


def test_ray_storage():
    assert "RAY_STORAGE" in os.environ


def test_ray_runtime_env():
    ctx = ray.get_runtime_context()
    env_vars = ctx.runtime_env.env_vars()
    assert env_vars["WANDB_MODE"] == "disabled"
