import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import PeftConfig
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    HuggingFaceRepoConfig,
    QuantizationConfig,
    TextDatasetConfig,
)
from flamingo.integrations.wandb import WandbArtifactConfig, get_artifact_filesystem_path


def _resolve_path_and_revision(
    load_from: HuggingFaceRepoConfig | WandbArtifactConfig,
) -> (str, str | None):
    match load_from:
        case HuggingFaceRepoConfig(repo_id, revision):
            model_path, revision = repo_id, revision
        case WandbArtifactConfig() as artifact_config:
            model_path = get_artifact_filesystem_path(artifact_config)
            revision = None
    return str(model_path), revision


def load_pretrained_config(config: AutoModelConfig) -> PretrainedConfig:
    model_path, revision = _resolve_path_and_revision(config.load_from)
    return AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
    )


def load_pretrained_model(
    config: AutoModelConfig,
    quantization: QuantizationConfig | None = None,
) -> PreTrainedModel:
    device_map, bnb_config = None, None
    if quantization is not None:
        bnb_config = quantization.as_huggingface()
        # When quantization is enabled, model must all be on same GPU to work with DDP
        # If a device_map is not specified we will get accelerate errors downstream
        # Reference: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
        current_device = Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
        device_map = {"": current_device}
        print(f"Setting model device_map = {device_map} to enable quantization")

    model_path, revision = _resolve_path_and_revision(config.load_from)
    return AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )


def load_pretrained_tokenizer(config: AutoTokenizerConfig) -> PreTrainedTokenizer:
    tokenizer_path, revision = _resolve_path_and_revision(config.load_from)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast,
    )


def load_peft_config(config: AutoModelConfig) -> PeftConfig:
    model_path, revision = _resolve_path_and_revision(config.load_from)
    pass


def load_dataset_from_config(config: TextDatasetConfig):
    dataset_path, revision = _resolve_path_and_revision(config.load_from)
    pass


def load_and_split_dataset(
    path: str,
    *,
    split: str | None = None,
    test_size: float | None,
    seed: int | None = None,
) -> DatasetDict:
    """Load a HuggingFace dataset and optionally perform a train/test split."""
    dataset = load_dataset(path, split=split)
    if test_size is not None:
        datasets = dataset.train_test_split(test_size=test_size, seed=seed)
    else:
        datasets = DatasetDict({"train": dataset})
    return datasets
