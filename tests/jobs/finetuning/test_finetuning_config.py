import pytest
from pydantic import ValidationError

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    TextDatasetConfig,
)
from flamingo.jobs.finetuning import FinetuningJobConfig, FinetuningRayConfig


@pytest.fixture
def finetuning_ray_config():
    return FinetuningRayConfig(
        num_workers=4,
        use_gpu=True,
    )


@pytest.fixture
def finetuning_job_config(
    model_config_with_artifact,
    dataset_config_with_artifact,
    tokenizer_config_with_artifact,
    quantization_config,
    lora_config,
    wandb_run_config,
    finetuning_ray_config,
):
    return FinetuningJobConfig(
        model=model_config_with_artifact,
        dataset=dataset_config_with_artifact,
        tokenizer=tokenizer_config_with_artifact,
        quantization=quantization_config,
        adapter=lora_config,
        tracking=wandb_run_config,
        ray=finetuning_ray_config,
    )


def test_serde_round_trip(finetuning_job_config):
    assert FinetuningJobConfig.parse_raw(finetuning_job_config.json()) == finetuning_job_config


def test_parse_yaml_file(finetuning_job_config, tmp_path_factory):
    config_path = tmp_path_factory.mktemp("flamingo_tests") / "finetuning_config.yaml"
    finetuning_job_config.to_yaml_file(config_path)
    assert finetuning_job_config == FinetuningJobConfig.from_yaml_file(config_path)


def test_load_example_config(examples_folder):
    """Load the example configs to make sure they stay up to date."""
    config_file = examples_folder / "configs" / "finetuning_config.yaml"
    config = FinetuningJobConfig.from_yaml_file(config_file)
    assert FinetuningJobConfig.parse_raw(config.json()) == config


def test_argument_validation():
    # Strings should be upcast to configs as the path argument
    allowed_config = FinetuningJobConfig(
        model="model_path",
        tokenizer="tokenizer_path",
        dataset="dataset_path",
    )
    assert allowed_config.model == AutoModelConfig(path="model_path")
    assert allowed_config.tokenizer == AutoTokenizerConfig(path="tokenizer_path")
    assert allowed_config.dataset == TextDatasetConfig(path="dataset_path")

    # Check passing invalid arguments is validated for each asset type
    with pytest.raises(ValidationError):
        FinetuningJobConfig(model=12345, tokenizer="tokenizer_path", dataset="dataset_path")
    with pytest.raises(ValidationError):
        FinetuningJobConfig(model="model_path", tokenizer=12345, dataset="dataset_path")
    with pytest.raises(ValidationError):
        FinetuningJobConfig(model="model_path", tokenizer="tokenizer_path", dataset=12345)

    # Check that tokenizer is set to model path when absent
    missing_tokenizer_config = FinetuningJobConfig(
        model="model_path",
        dataset="dataset_path",
    )
    assert missing_tokenizer_config.tokenizer.path == "model_path"
