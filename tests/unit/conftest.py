import pytest
from peft import PeftType, TaskType

from flamingo.integrations.huggingface import (
    AdapterConfig,
    AutoModelConfig,
    AutoTokenizerConfig,
    QuantizationConfig,
    TextDatasetConfig,
)
from flamingo.integrations.vllm import InferenceServerConfig
from flamingo.integrations.wandb import WandbArtifactConfig, WandbRunConfig


@pytest.fixture
def model_config_with_repo_id():
    return AutoModelConfig(load_from="mistral-ai/mistral-7", trust_remote_code=True)


@pytest.fixture
def model_config_with_artifact():
    artifact = WandbArtifactConfig(name="model", project="project")
    return AutoModelConfig(load_from=artifact, trust_remote_code=True)


@pytest.fixture
def inference_server_config():
    return InferenceServerConfig(base_url="1.2.3.4:8000/v1/completions")


@pytest.fixture
def tokenizer_config_with_repo_id():
    return AutoTokenizerConfig(load_from="mistral-ai/mistral-7", trust_remote_code=True)


@pytest.fixture
def tokenizer_config_with_artifact():
    artifact = WandbArtifactConfig(name="tokenizer", project="project")
    return AutoTokenizerConfig(load_from=artifact, trust_remote_code=True)


@pytest.fixture
def dataset_config_with_repo_id():
    return TextDatasetConfig(load_from="databricks/dolly15k", text_field="text", split="train")


@pytest.fixture
def dataset_config_with_artifact():
    artifact = WandbArtifactConfig(name="dataset", project="project")
    return TextDatasetConfig(load_from=artifact, split="train")


@pytest.fixture
def quantization_config():
    return QuantizationConfig(load_in_8bit=True)


@pytest.fixture
def adapter_config():
    return AdapterConfig(
        peft_type=PeftType.LORA,
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.2,
    )


@pytest.fixture
def wandb_run_config():
    return WandbRunConfig(name="run", run_id="12345", project="research", entity="mzai")
