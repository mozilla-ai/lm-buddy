import ray
from ray.train import CheckpointConfig, DataConfig, RunConfig, ScalingConfig, get_dataset_shard
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
    WandbResumeMode,
    default_artifact_name,
    log_directory_reference,
    wandb_init_from_config,
)
from flamingo.jobs.finetuning import FinetuningJobConfig
from flamingo.jobs.finetuning.utils import generate_huggingface_dataset, is_tracking_enabled
from flamingo.jobs.utils import FlamingoJobType


def load_and_train(config: FinetuningJobConfig):
    # Load the model and tokenizer
    model = load_pretrained_model(config.model, config.quantization)
    tokenizer = load_pretrained_tokenizer(config.tokenizer)

    # Get the Ray dataset shards and convert back to HuggingFace format
    # The SFTTrainer handles pre-processing internally (e.g., tokenization, batch collation),
    # but only when provided a `datasets.Dataset` instance
    # TODO: Is there a smarter/more efficient way to do this with the re-conversion?
    train_ds = get_dataset_shard("train")
    train_ds = generate_huggingface_dataset(train_ds)

    eval_ds = get_dataset_shard("test")
    eval_ds = generate_huggingface_dataset(eval_ds) if eval_ds else None

    print(f"Train Dataset: {train_ds}")
    print(f"Eval Dataset: {eval_ds}")

    training_args = TrainingArguments(
        output_dir="out",  # Local checkpoint path on a worker
        report_to="wandb" if config.tracking and is_tracking_enabled() else "none",
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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=config.trainer.max_seq_length,
        dataset_text_field=config.dataset.text_field,
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()


def training_function(config_data: dict):
    config = FinetuningJobConfig(**config_data)
    if config.tracking and is_tracking_enabled():
        with wandb_init_from_config(
            config.tracking, resume=WandbResumeMode.NEVER, job_type=FlamingoJobType.FINETUNING
        ):
            load_and_train(config)
    else:
        load_and_train(config)


def run_finetuning(config: FinetuningJobConfig):
    # Load data and convert to Ray datasets
    # FIXME: This is not logging input artifacts, fix in the next dev PR
    hf_datasets = load_and_split_dataset(config.dataset)
    ray_datasets = {split: ray.data.from_huggingface(d) for split, d in hf_datasets.items()}

    # Construct Ray train configurations from input config
    data_config = DataConfig(
        datasets_to_split=config.ray.datasets_to_shard,
    )
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
        train_loop_config=config.dict(),
        scaling_config=scaling_config,
        run_config=run_config,
        dataset_config=data_config,
        datasets=ray_datasets,
    )

    # Fit the trainer
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
