import pytest

from flamingo.jobs.lm_harness import (
    LMHarnessEvaluatorConfig,
    LMHarnessJobConfig,
    LMHarnessRayConfig,
)


@pytest.fixture
def lm_harness_evaluator_config():
    return LMHarnessEvaluatorConfig(
        tasks=["task1", "task2", "task3"],
        num_fewshot=5,
    )


@pytest.fixture
def lm_harness_ray_config():
    return LMHarnessRayConfig(
        num_cpus=2,
        num_gpus=4,
        timeout=3600,
    )


@pytest.fixture
def lm_harness_job_config(
    request,
    model_config_with_artifact,
    inference_server_config,
    quantization_config,
    wandb_run_config,
    lm_harness_evaluator_config,
    lm_harness_ray_config,
):
    if request.param == "model_config_with_artifact":
        return LMHarnessJobConfig(
            model=model_config_with_artifact,
            evaluator=lm_harness_evaluator_config,
            ray=lm_harness_ray_config,
            tracking=wandb_run_config,
            quantization=quantization_config,
        )
    elif request.param == "inference_server_config":
        return LMHarnessJobConfig(
            model=inference_server_config,
            evaluator=lm_harness_evaluator_config,
            ray=lm_harness_ray_config,
            tracking=wandb_run_config,
            quantization=quantization_config,
        )


@pytest.mark.parametrize(
    "lm_harness_job_config",
    ["model_config_with_artifact", "inference_server_config"],
    indirect=True,
)
def test_serde_round_trip(lm_harness_job_config):
    assert LMHarnessJobConfig.parse_raw(lm_harness_job_config.json()) == lm_harness_job_config


@pytest.mark.parametrize(
    "lm_harness_job_config",
    ["model_config_with_artifact", "inference_server_config"],
    indirect=True,
)
def test_parse_yaml_file(lm_harness_job_config):
    with lm_harness_job_config.to_tempfile() as config_path:
        assert lm_harness_job_config == LMHarnessJobConfig.from_yaml_file(config_path)


@pytest.mark.parametrize(
    "file_suffix", ["lm_harness_hf_config.yaml", "lm_harness_inference_server_config.yaml"]
)
def test_load_example_config(examples_dir, file_suffix):
    """Load the example configs to make sure they stay up to date."""
    config_file = examples_dir / "configs" / file_suffix
    config = LMHarnessJobConfig.from_yaml_file(config_file)
    assert LMHarnessJobConfig.parse_raw(config.json()) == config
