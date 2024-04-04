from pathlib import Path
from typing import Any

import wandb
from ray import train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import TrainingArguments
from trl import SFTTrainer

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import (
    ArtifactType,
    WandbResumeMode,
    build_directory_artifact,
    default_artifact_name,
)
from lm_buddy.jobs.common import FinetuningResult, LMBuddyJobType, preprocess_text_dataset
from lm_buddy.jobs.configs import FinetuningJobConfig


def is_tracking_enabled(config: FinetuningJobConfig):
    # Only report to WandB on the rank 0 worker
    # Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    return config.tracking is not None and train.get_context().get_world_rank() == 0


def load_and_train(config: FinetuningJobConfig):
    # Load the HF assets from configurations
    hf_loader = HuggingFaceAssetLoader()
    model = hf_loader.load_pretrained_model(config.model, config.quantization)
    tokenizer = hf_loader.load_pretrained_tokenizer(config.tokenizer)

    datasets = hf_loader.load_and_split_dataset(config.dataset)
    for split, dataset in datasets.items():
        datasets[split] = preprocess_text_dataset(dataset, config.dataset)

    training_args = TrainingArguments(
        output_dir="out",  # Local checkpoint path on a worker
        report_to="wandb" if is_tracking_enabled(config) else "none",
        use_cpu=not config.ray.use_gpu,
        push_to_hub=False,
        disable_tqdm=True,
        logging_dir=None,
        **config.trainer.training_args(),
    )

    peft_config = config.adapter.as_huggingface() if config.adapter else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
        max_seq_length=config.trainer.max_seq_length,
        dataset_text_field=config.dataset.text_field,
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()


def training_function(config_data: dict[str, Any]):
    config = FinetuningJobConfig(**config_data)
    if is_tracking_enabled(config):
        with wandb.init(
            name=config.name,
            resume=WandbResumeMode.NEVER,
            job_type=LMBuddyJobType.FINETUNING,
            **config.tracking.model_dump(),
        ):
            load_and_train(config)
    else:
        load_and_train(config)


def run_finetuning(config: FinetuningJobConfig) -> FinetuningResult:
    # Construct Ray train configurations from input config
    scaling_config = ScalingConfig(
        use_gpu=config.ray.use_gpu,
        num_workers=config.ray.num_workers,
    )
    run_config = RunConfig(
        name=config.name,
        storage_path=config.ray.storage_path,
        checkpoint_config=CheckpointConfig(num_to_keep=1),
    )
    trainer = TorchTrainer(
        train_loop_per_worker=training_function,
        train_loop_config=config.model_dump(),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"Training result: {result}")

    # Create a checkpoint artifact if tracking is enabled and Ray saved a checkpoint
    if result.checkpoint:
        checkpoint_path = Path(result.checkpoint.path) / RayTrainReportCallback.CHECKPOINT_NAME
        checkpoint_artifact = build_directory_artifact(
            artifact_name=default_artifact_name(config.name, ArtifactType.MODEL),
            artifact_type=ArtifactType.MODEL,
            dir_path=checkpoint_path,
            reference=True,
        )
    else:
        checkpoint_path, checkpoint_artifact = None, None

    # Return finetuning result object
    return FinetuningResult(
        artifacts=[checkpoint_artifact] if checkpoint_artifact else [],
        checkpoint_path=checkpoint_path,
        metrics=result.metrics,
        is_adapter=config.adapter is not None,
    )
