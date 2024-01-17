import pytest
from pydantic import ValidationError

from flamingo.integrations.huggingface import AutoModelConfig
from flamingo.jobs import LMHarnessJobConfig
from flamingo.jobs.lm_harness_config import LMHarnessEvaluatorConfig, LMHarnessRayConfig


@pytest.fixture
def lm_harness_evaluator_config():
    return LMHarnessEvaluatorConfig(
        tasks=["task1", "task2", "task3"],
        num_fewshot=5,
    )


@pytest.fixture
def lm_harness_ray_config():
    return LMHarnessRayConfig(
        num_workers=4,
        use_gpu=True,
    )


@pytest.fixture
def lm_harness_job_config(
    model_config_with_artifact,
    quantization_config,
    wandb_run_config,
    lm_harness_evaluator_config,
    lm_harness_ray_config,
):
    return LMHarnessJobConfig(
        model=model_config_with_artifact,
        evaluator=lm_harness_evaluator_config,
        ray=lm_harness_ray_config,
        tracking=wandb_run_config,
        quantization=quantization_config,
    )


def test_serde_round_trip(lm_harness_job_config):
    assert LMHarnessJobConfig.parse_raw(lm_harness_job_config.json()) == lm_harness_job_config


def test_parse_yaml_file(lm_harness_job_config, tmp_path_factory):
    config_path = tmp_path_factory.mktemp("flamingo_tests") / "lm_harness_config.yaml"
    lm_harness_job_config.to_yaml_file(config_path)
    assert lm_harness_job_config == LMHarnessJobConfig.from_yaml_file(config_path)


def test_model_validation(lm_harness_evaluator_config):
    allowed_config = LMHarnessJobConfig(model="hf_repo_id", evaluator=lm_harness_evaluator_config)
    assert allowed_config.model == AutoModelConfig(path="hf_repo_id")

    with pytest.raises(ValidationError):
        LMHarnessJobConfig(model="invalid...hf..repo", evaluator=lm_harness_evaluator_config)

    with pytest.raises(ValidationError):
        LMHarnessJobConfig(model=12345, evaluator=lm_harness_evaluator_config)
