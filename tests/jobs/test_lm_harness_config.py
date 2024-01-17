import pytest
from pydantic import ValidationError

from flamingo.integrations.huggingface import AutoModelConfig
from flamingo.jobs import LMHarnessJobConfig
from flamingo.jobs.lm_harness_config import LMHarnessEvaluatorConfig, LMHarnessRayConfig
from tests.conftest import TEST_RESOURCES


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


def test_model_validation(lm_harness_evaluator_config):
    allowed_config = LMHarnessJobConfig(model="hf_repo_id", evaluator=lm_harness_evaluator_config)
    assert allowed_config.model == AutoModelConfig(path="hf_repo_id")

    with pytest.raises(ValidationError):
        LMHarnessJobConfig(model="invalid...hf..repo", evaluator=lm_harness_evaluator_config)

    with pytest.raises(ValidationError):
        LMHarnessJobConfig(model=12345, evaluator=lm_harness_evaluator_config)


def test_serde_round_trip(
    model_config_with_artifact,
    quantization_config,
    wandb_run_config,
    lm_harness_evaluator_config,
    lm_harness_ray_config,
):
    config = LMHarnessJobConfig(
        model=model_config_with_artifact,
        evaluator=lm_harness_evaluator_config,
        ray=lm_harness_ray_config,
        tracking=wandb_run_config,
        quantization=quantization_config,
    )
    assert LMHarnessJobConfig.parse_raw(config.json()) == config


def test_parse_yaml_file(tmp_path_factory):
    load_path = TEST_RESOURCES / "lm_harness_config.yaml"
    config = LMHarnessJobConfig.from_yaml_file(load_path)
    write_path = tmp_path_factory.mktemp("flamingo_tests") / "harness_config.yaml"
    config.to_yaml_file(write_path)
    assert config == LMHarnessJobConfig.from_yaml_file(write_path)
