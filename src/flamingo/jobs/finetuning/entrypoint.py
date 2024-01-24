import json

from ray import train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import TrainingArguments
from trl import SFTTrainer

from flamingo.integrations.huggingface import (
    load_and_split_dataset,
    load_pretrained_model,
    load_pretrained_tokenizer,
)
from flamingo.integrations.wandb import (
    ArtifactType,
    ArtifactURIScheme,
    default_artifact_name,
    log_directory_reference,
    wandb_init_from_config,
)
from flamingo.jobs.finetuning import FinetuningJobConfig
from flamingo.jobs.utils import FlamingoJobType


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


def load_and_train(config: FinetuningJobConfig):
    # Load the input artifacts, potentially linking them to the active W&B run
    model = load_pretrained_model(config.model, config.quantization)
    tokenizer = load_pretrained_tokenizer(config.tokenizer)
    datasets = load_and_split_dataset(config.dataset)

    training_args = get_training_arguments(config)
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


def training_function(config_data: dict):
    config = FinetuningJobConfig(**config_data)
    if is_tracking_enabled(config):
        with wandb_init_from_config(
            config.tracking,
            job_type=FlamingoJobType.FINETUNING,
            resume="never",
        ):
            load_and_train(config)
    else:
        load_and_train(config)


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
        train_loop_per_worker=training_function,
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
            print("Logging artifact for model checkpoint...")
            log_directory_reference(
                dir_path=f"{result.checkpoint.path}/checkpoint",
                artifact_name=default_artifact_name(run.name, ArtifactType.MODEL),
                artifact_type=ArtifactType.MODEL,
                scheme=ArtifactURIScheme.FILE,
            )
