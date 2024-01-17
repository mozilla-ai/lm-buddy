import json

import torch
import wandb
from accelerate import Accelerator
from datasets import DatasetDict
from ray import train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, TrainingArguments
from trl import SFTTrainer

from flamingo.integrations.huggingface.utils import load_and_split_dataset
from flamingo.integrations.wandb import WandbArtifactConfig, WandbArtifactLoader
from flamingo.integrations.wandb.utils import get_reference_filesystem_path
from flamingo.jobs import FinetuningJobConfig


def is_tracking_enabled(config: FinetuningJobConfig):
    # Only report to WandB on the rank 0 worker
    # Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    return config.tracking is not None and train.get_context().get_world_rank() == 0


def resolve_artifact_path(path: str | WandbArtifactConfig, loader: WandbArtifactLoader) -> str:
    """Resolve the actual filesystem path for a path/artifact asset.

    The artifact loader internally handles linking the artifact-to-load to an in-progress run.
    """
    match path:
        case str():
            return path
        case WandbArtifactConfig() as artifact_config:
            artifact = loader.load_artifact(artifact_config)
            return get_reference_filesystem_path(artifact)
        case _:
            raise ValueError(f"Invalid artifact path: {path}")


def get_training_arguments(config: FinetuningJobConfig) -> TrainingArguments:
    """Get TrainingArguments appropriate for the worker rank and job config."""
    provided_args = config.trainer.get_training_args()
    return TrainingArguments(
        output_dir="out",  # Local checkpoint path on a worker
        report_to="wandb" if is_tracking_enabled(config) else "none",
        no_cuda=not config.scaling.use_gpu,
        push_to_hub=False,
        disable_tqdm=True,
        logging_dir=None,
        **provided_args,
    )


def load_datasets(config: FinetuningJobConfig, loader: WandbArtifactLoader) -> DatasetDict:
    dataset_path = resolve_artifact_path(config.dataset.path, loader)
    # We need to specify a fixed seed to load the datasets on each worker
    # Under the hood, HuggingFace uses `accelerate` to create a data loader shard for each worker
    # If the datasets are not seeded here, the ordering will be inconsistent between workers
    # TODO: Get rid of this logic once data loading occurs once outside of the workers
    split_seed = config.dataset.seed or 0
    return load_and_split_dataset(
        dataset_path,
        split=config.dataset.split,
        test_size=config.dataset.test_size,
        seed=split_seed,
    )


def load_model(config: FinetuningJobConfig, loader: WandbArtifactLoader) -> PreTrainedModel:
    device_map, bnb_config = None, None
    if config.quantization is not None:
        bnb_config = config.quantization.as_huggingface()
        # When quantization is enabled, model must all be on same GPU to work with DDP
        # If a device_map is not specified we will get accelerate errors downstream
        # Reference: https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
        current_device = Accelerator().local_process_index if torch.cuda.is_available() else "cpu"
        device_map = {"": current_device}
        print(f"Setting model device_map = {device_map} to enable quantization")

    model_path = resolve_artifact_path(config.model.path, loader)
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=config.model.torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )


def load_tokenizer(config: FinetuningJobConfig, loader: WandbArtifactLoader):
    tokenizer_path = resolve_artifact_path(config.tokenizer.path, loader)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        trust_remote_code=config.tokenizer.trust_remote_code,
        use_fast=config.tokenizer.use_fast,
    )
    if not tokenizer.pad_token_id:
        # Pad token required for generating consistent batch sizes
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def train_func(config_data: dict):
    config = FinetuningJobConfig(**config_data)
    training_args = get_training_arguments(config)

    # Manually initialize run in order to set the run ID and link artifacts
    wandb_run = None
    if is_tracking_enabled(config):
        env = config.tracking
        wandb_run = wandb.init(
            id=env.run_id,
            name=env.name,
            project=env.project,
            entity=env.entity,
            group=env.run_group,
        )

    # Load the input artifacts, potentially linking them to the active W&B run
    artifact_loader = WandbArtifactLoader(wandb_run)
    datasets = load_datasets(config, artifact_loader)
    model = load_model(config, artifact_loader)
    tokenizer = load_tokenizer(config, artifact_loader)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=config.adapter,
        max_seq_length=config.trainer.max_seq_length,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
        dataset_text_field="text",
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

    # Force WandB finish on rank 0 worker
    if is_tracking_enabled(config):
        wandb.finish()


def run_finetuning(config: FinetuningJobConfig):
    print(f"Received job configuration: {config}")

    scaling_config = ScalingConfig(**config.ray.get_scaling_args())
    run_config = RunConfig(
        name=config.tracking.name if config.tracking else None,
        storage_path=config.ray.storage_path,
        checkpoint_config=CheckpointConfig(num_to_keep=1),
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=json.loads(config.json()),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"Training result: {result}")

    if config.tracking:
        # TODO: Add ref artifact here
        pass
