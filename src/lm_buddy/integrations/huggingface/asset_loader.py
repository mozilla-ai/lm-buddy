import warnings

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import PeftConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from lm_buddy.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    DatasetConfig,
    QuantizationConfig,
)
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    get_artifact_directory,
)
from lm_buddy.paths import AssetPath, FilePath, HuggingFacePath, PathPrefix, strip_path_prefix


def resolve_peft_and_pretrained(
    path: FilePath | HuggingFacePath,
) -> tuple[str, str | None]:
    """Helper method for determining if a path corresponds to a PEFT model.

    A PEFT model contains an `adapter_config.json` in its directory.
    If this file can be loaded, we know the path is a for a PEFT model.
    If not, we assume the provided path corresponds to a base HF model.

    Args:
        path (str): Name/path to a HuggingFace directory

    Returns:
        Tuple of (base model path, optional PEFT path)
    """
    # We don't know if the checkpoint is adapter weights or merged model weights
    # Try to load as an adapter and fall back to the checkpoint containing the full model
    try:
        path = strip_path_prefix(path)
        peft_config = PeftConfig.from_pretrained(path)
        return peft_config.base_model_name_or_path, path
    except ValueError as e:
        warnings.warn(
            f"Unable to load model as adapter: {e}. "
            "This is expected if the checkpoint does not contain adapter weights."
        )
        return path, None


class HuggingFaceAssetLoader:
    """Helper class for loading HuggingFace assets from LM Buddy configurations.

    This class depends on an `ArtifactLoader` in order to resolve actual paths from
    artifact references.
    """

    def __init__(self, artifact_loader: ArtifactLoader):
        self._artifact_loader = artifact_loader

    def resolve_asset_path(self, path: AssetPath) -> FilePath | HuggingFacePath:
        """Resolve the loadable version of an `AssetPath`.

        W&B paths are resolved to file paths given the artifact manifest.
        The returned path contains the `PathPrefix`.
        """
        if path.startswith((PathPrefix.FILE, PathPrefix.HUGGINGFACE)):
            return path
        elif path.startswith(PathPrefix.WANDB):
            artifact = self._artifact_loader.use_artifact(path)
            return get_artifact_directory(artifact)
        else:
            raise ValueError(f"Unable to resolve asset path from {path}.")

    def load_pretrained_config(
        self,
        config: AutoModelConfig,
    ) -> PretrainedConfig:
        """Load a `PretrainedConfig` from the model configuration.

        An exception is raised if the HuggingFace repo does not contain a `config.json` file.
        """
        config_path = self.resolve_asset_path(config.path)
        config_path = strip_path_prefix(config_path)
        return AutoConfig.from_pretrained(pretrained_model_name_or_path=config_path)

    def load_pretrained_model(
        self,
        config: AutoModelConfig,
        quantization: QuantizationConfig | None = None,
    ) -> PreTrainedModel:
        """Load a `PreTrainedModel` with optional quantization from the model configuration.

        An exception is raised if the HuggingFace repo does not contain a `config.json` file.

        TODO(RD2024-87): This fails if the checkpoint only contains a PEFT adapter config
        """
        device_map, bnb_config = None, None
        if quantization is not None:
            bnb_config = quantization.as_huggingface()
            # When quantization is enabled, model must all be on same GPU to work with DDP
            # If a device_map is not specified we will get accelerate errors downstream
            # Reference: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
            current_device = (
                Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
            )
            device_map = {"": current_device}
            print(f"Setting model device_map = {device_map} to enable quantization")

        # TODO: HuggingFace has many AutoModel classes with different "language model heads"
        #   Can we abstract this to load with any type of AutoModel class?
        model_path = self.resolve_asset_path(config.path)
        model_path = strip_path_prefix(model_path)
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=config.torch_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
        )

    def load_pretrained_tokenizer(self, config: AutoTokenizerConfig) -> PreTrainedTokenizer:
        """Load a `PreTrainedTokenizer` from the model configuration.

        An exception is raised if the HuggingFace repo does not contain a `tokenizer.json` file.
        """
        tokenizer_path = self.resolve_asset_path(config.path)
        tokenizer_path = strip_path_prefix(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=config.use_fast,
        )
        if tokenizer.pad_token_id is None:
            # Pad token required for generating consistent batch sizes
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load_dataset(self, config: DatasetConfig) -> Dataset:
        """Load a HuggingFace `Dataset` from the dataset configuration.

        This method always returns a single `Dataset` object.
        When loading from HuggingFace directly, the `Dataset` is for the provided split.
        When loading from disk, the saved files must be for a dataset else an exception is raised.
        """
        dataset_path = self.resolve_asset_path(config.path)
        # Dataset loading requires a different method if from a HF vs. disk
        if dataset_path.startswith(PathPrefix.HUGGINGFACE):
            return load_dataset(strip_path_prefix(dataset_path), split=config.split)
        else:
            match load_from_disk(strip_path_prefix(dataset_path)):
                case Dataset() as dataset:
                    return dataset
                case other:
                    raise ValueError(
                        "LM Buddy currently only supports loading `Dataset` objects from disk, "
                        f"instead found a {type(other)}."
                    )

    def load_and_split_dataset(self, config: DatasetConfig) -> DatasetDict:
        """Load a HuggingFace dataset and optionally perform a train/test split.

        The split is performed when a `test_size` is specified on the configuration.
        """
        match self.load_dataset(config):
            case Dataset() as dataset if config.test_size is not None:
                # We need to specify a fixed seed to load the datasets on each worker
                # Under the hood, HuggingFace uses `accelerate` to create a data loader shards
                # If the datasets are not seeded here, the ordering will be inconsistent
                # TODO: Get rid of this when data is loaded once outside of Ray workers
                split_seed = config.seed or 0
                return dataset.train_test_split(test_size=config.test_size, seed=split_seed)
            case dataset:
                return DatasetDict({"train": dataset})
