from typing import Any

import ray
from ray import train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import TrainingArguments
from trl import SFTTrainer

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    ArtifactType,
    WandbResumeMode,
    build_directory_artifact,
    default_artifact_name,
    wandb_init_from_config,
)
from lm_buddy.jobs.common import LMBuddyJobType
from lm_buddy.jobs.configs import FinetuningJobConfig


def is_tracking_enabled(config: FinetuningJobConfig):
    # Only report to WandB on the rank 0 worker
    # Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    return config.tracking is not None and train.get_context().get_world_rank() == 0


def load_and_train(config: FinetuningJobConfig, artifact_loader: ArtifactLoader):
    # Load the HF assets from configurations
    # Internally, artifact lineages are declared for the active training run
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    model = hf_loader.load_pretrained_model(config.model, config.quantization)
    tokenizer = hf_loader.load_pretrained_tokenizer(config.tokenizer)
    datasets = hf_loader.load_and_split_dataset(config.dataset)

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


def run_finetuning(config: FinetuningJobConfig, artifact_loader: ArtifactLoader):
    # Place the artifact loader in Ray object store
    artifact_loader_ref = ray.put(artifact_loader)

    # Define training function internally to capture the artifact loader ref as a closure
    # Reference: https://docs.ray.io/en/latest/ray-core/objects.html#closure-capture-of-objects
    def training_function(config_data: dict[str, Any]):
        artifact_loader = ray.get(artifact_loader_ref)
        config = FinetuningJobConfig(**config_data)
        if is_tracking_enabled(config):
            with wandb_init_from_config(
                config.tracking,
                resume=WandbResumeMode.NEVER,
                job_type=LMBuddyJobType.FINETUNING,
            ):
                load_and_train(config, artifact_loader)
        else:
            load_and_train(config, artifact_loader)

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
        train_loop_per_worker=training_function,
        train_loop_config=config.model_dump(),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"Training result: {result}")

    # Register a model artifact if tracking is enabled and Ray saved a checkpoint
    if config.tracking and result.checkpoint:
        # Must resume from the just-completed training run
        with wandb_init_from_config(config.tracking, resume=WandbResumeMode.MUST) as run:
            model_artifact = build_directory_artifact(
                artifact_name=default_artifact_name(run.name, ArtifactType.MODEL),
                artifact_type=ArtifactType.MODEL,
                dir_path=f"{result.checkpoint.path}/{RayTrainReportCallback.CHECKPOINT_NAME}",
                reference=True,
            )
            print("Logging artifact for model checkpoint...")
            artifact_loader.log_artifact(model_artifact)
