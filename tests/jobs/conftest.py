import pytest
from peft import LoraConfig

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    QuantizationConfig,
    TextDatasetConfig,
)
from flamingo.integrations.wandb import WandbArtifactConfig, WandbRunConfig


@pytest.fixture
def model_config_with_path():
    return AutoModelConfig(load_from="mistral-ai/mistral-7", trust_remote_code=True)


@pytest.fixture
def model_config_with_artifact():
    artifact = WandbArtifactConfig(name="model")
    return AutoModelConfig(load_from=artifact, trust_remote_code=True)


@pytest.fixture
def tokenizer_config_with_path():
    return AutoTokenizerConfig(load_from="mistral-ai/mistral-7", trust_remote_code=True)


@pytest.fixture
def tokenizer_config_with_artifact():
    artifact = WandbArtifactConfig(name="tokenizer")
    return AutoTokenizerConfig(load_from=artifact, trust_remote_code=True)


@pytest.fixture
def dataset_config_with_path():
    return TextDatasetConfig(
        load_from="databricks/dolly15k",
        text_field="text",
        split="train",
    )


@pytest.fixture
def dataset_config_with_artifact():
    artifact = WandbArtifactConfig(name="dataset")
    return TextDatasetConfig(load_from=artifact, split="train")


@pytest.fixture
def quantization_config():
    return QuantizationConfig(load_in_8bit=True)


@pytest.fixture
def lora_config():
    return LoraConfig(r=8, lora_alpha=32, lora_dropout=0.2)


@pytest.fixture
def wandb_run_config():
    return WandbRunConfig(name="run", run_id="12345", project="research", entity="mozilla-ai")
