import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    HuggingFaceRepoConfig,
    LoadFromConfig,
    QuantizationConfig,
)
from flamingo.integrations.huggingface.dataset_config import DatasetConfig
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.integrations.wandb.artifact_utils import get_artifact_filesystem_path


def resolve_loadable_path(load_from: LoadFromConfig) -> (str, str | None):
    """Resolve the loadable path and revision from configuration.

    If a `HuggingFaceRepoConfig` is provided, return the values directly.
    If a `WandbArtifactConfig` is provided, resolve the path from the artifact manifest.
    """
    match load_from:
        case HuggingFaceRepoConfig(repo_id, revision):
            load_path, revision = repo_id, revision
        case WandbArtifactConfig() as artifact_config:
            load_path = get_artifact_filesystem_path(artifact_config)
            revision = None
        case _:
            raise ValueError(f"Unable to resolve load path from {load_from}.")
    return str(load_path), revision


def load_pretrained_model_config(config: AutoModelConfig) -> PretrainedConfig:
    """Load a `PretrainedConfig` from the flamingo configuration.

    An exception is raised if the HuggingFace repo does not contain a `config.json` file.
    """
    model_path, revision = resolve_loadable_path(config.load_from)
    return AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, revision=revision)


def load_pretrained_model(
    config: AutoModelConfig,
    quantization: QuantizationConfig | None = None,
) -> PreTrainedModel:
    """Load a `PreTrainedModel` with optional quantization from the flamingo configuration.

    An exception is raised if the HuggingFace repo does not contain a `config.json` file.
    """
    device_map, bnb_config = None, None
    if quantization is not None:
        bnb_config = quantization.as_huggingface()
        # When quantization is enabled, model must all be on same GPU to work with DDP
        # If a device_map is not specified we will get accelerate errors downstream
        # Reference: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
        current_device = Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
        device_map = {"": current_device}
        print(f"Setting model device_map = {device_map} to enable quantization")

    # TODO: HuggingFace has many AutoModel classes with different "language model heads"
    #   Can we abstract this to load with any type of AutoModel class?
    model_path, revision = resolve_loadable_path(config.load_from)
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )


def load_pretrained_tokenizer(config: AutoTokenizerConfig) -> PreTrainedTokenizer:
    """Load a `PreTrainedTokenizer` from the flamingo configuration.

    An exception is raised if the HuggingFace repo does not contain a `tokenizer.json` file.
    """
    tokenizer_path, revision = resolve_loadable_path(config.load_from)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        revision=revision,
        trust_remote_code=config.trust_remote_code,
        use_fast=config.use_fast,
    )
    if tokenizer.pad_token_id is None:
        # Pad token required for generating consistent batch sizes
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_dataset_from_config(config: DatasetConfig) -> Dataset:
    """Load a HuggingFace `Dataset` from the flamingo configuration.

    This method always returns a single `Dataset` object.
    When loading from HuggingFace directly, the `Dataset` is for the provided split.
    When loading from disk, the saved files must be for a dataset else an exception is raised.
    """
    dataset_path, revision = resolve_loadable_path(config.load_from)
    # Dataset loading requires a different method if from a HF vs. disk
    if isinstance(config.load_from, HuggingFaceRepoConfig):
        return load_dataset(dataset_path, revision=revision, split=config.split)
    else:
        match load_from_disk(dataset_path):
            case Dataset() as dataset:
                return dataset
            case other:
                raise ValueError(
                    "Flamingo currently only supports loading `Dataset` objects from disk, "
                    f"instead found a {type(other)}."
                )


def load_and_split_dataset(config: DatasetConfig) -> DatasetDict:
    """Load a HuggingFace dataset and optionally perform a train/test split.

    The split is performed when a `test_size` is specified on the configuration.
    """
    match load_dataset_from_config(config):
        case Dataset() as dataset if config.test_size is not None:
            # We need to specify a fixed seed to load the datasets on each worker
            # Under the hood, HuggingFace uses `accelerate` to create a data loader shards
            # If the datasets are not seeded here, the ordering will be inconsistent
            # TODO: Get rid of this logic once data loading is done one time outside of Ray workers
            split_seed = config.seed or 0
            return dataset.train_test_split(test_size=config.test_size, seed=split_seed)
        case dataset:
            return DatasetDict({"train": dataset})
