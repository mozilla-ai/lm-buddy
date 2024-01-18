import json

import torch
import wandb
from accelerate import Accelerator
from datasets import DatasetDict
from ray import train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, TrainingArguments
from trl import SFTTrainer

from flamingo.integrations.huggingface.utils import load_and_split_dataset
from flamingo.integrations.wandb import ArtifactType, WandbArtifactLoader
from flamingo.integrations.wandb.utils import default_artifact_name, wandb_init_from_config
from flamingo.jobs.finetuning import FinetuningJobConfig


def build_model_artifact(run_name: str, checkpoint: Checkpoint) -> wandb.Artifact:
    print("Building artifact for model checkpoint...")
    artifact_name = default_artifact_name(run_name, ArtifactType.MODEL)
    artifact = wandb.Artifact(artifact_name, type=ArtifactType.MODEL.value)
    artifact.add_reference(f"file://{checkpoint.path}/checkpoint")
    return artifact


def is_tracking_enabled(config: FinetuningJobConfig):
    # Only report to WandB on the rank 0 worker
    # Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    return config.tracking is not None and train.get_context().get_world_rank() == 0


def get_training_arguments(config: FinetuningJobConfig) -> TrainingArguments:
    """Get TrainingArguments appropriate for the worker rank and job config."""
    return TrainingArguments(
        output_dir="out",  # Local checkpoint path on a worker
        report_to="wandb" if is_tracking_enabled(config) else "none",
        no_cuda=not config.ray.use_gpu,
        push_to_hub=False,
        disable_tqdm=True,
        logging_dir=None,
        **config.trainer.training_args(),
    )


def load_datasets(config: FinetuningJobConfig, loader: WandbArtifactLoader) -> DatasetDict:
    dataset_path = loader.resolve_artifact_path(config.dataset.path)
    # We need to specify a fixed seed to load the datasets on each worker
    # Under the hood, HuggingFace uses `accelerate` to create a data loader shard for each worker
    # If the datasets are not seeded here, the ordering will be inconsistent between workers
    # TODO: Get rid of this logic once data loading is done one time outside of the workers
    split_seed = config.dataset.seed or 0
    return load_and_split_dataset(
        path=dataset_path,
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

    model_path = loader.resolve_artifact_path(config.model.path)
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=config.model.torch_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )


def load_tokenizer(config: FinetuningJobConfig, loader: WandbArtifactLoader):
    tokenizer_path = loader.resolve_artifact_path(config.tokenizer.path)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        trust_remote_code=config.tokenizer.trust_remote_code,
        use_fast=config.tokenizer.use_fast,
    )
    if not tokenizer.pad_token_id:
        # Pad token required for generating consistent batch sizes
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def train_func_with_loader(config: FinetuningJobConfig, loader: WandbArtifactLoader):
    training_args = get_training_arguments(config)

    # Load the input artifacts, potentially linking them to the active W&B run
    datasets = load_datasets(config, loader)
    model = load_model(config, loader)
    tokenizer = load_tokenizer(config, loader)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=config.adapter,
        max_seq_length=config.trainer.max_seq_length,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
        dataset_text_field=config.dataset.text_field,
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()


def train_func(config_data: dict):
    config = FinetuningJobConfig(**config_data)
    if is_tracking_enabled(config):
        with wandb_init_from_config(config, resume="never") as run:
            loader = WandbArtifactLoader(run=run)
            train_func_with_loader(config, loader)
    else:
        loader = WandbArtifactLoader(run=None)
        train_func_with_loader(config, loader)


def run_finetuning(config: FinetuningJobConfig):
    # Construct Ray train configurations from input config
    scaling_config = ScalingConfig(
        use_gpu=config.ray.use_gpu,
        num_workers=config.ray.num_workers,
    )
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

    # Register a model artifact if tracking is enabled and Ray saved a checkpoint
    if config.tracking and result.checkpoint:
        # Must resume from the just-completed training run
        with wandb_init_from_config(config.tracking, resume="must") as run:
            artifact = build_model_artifact(run.name, result.checkpoint)
            run.log_artifact(artifact)
