from pathlib import Path
from typing import Any

import ray
from ray import train
from ray.train import CheckpointConfig, Result, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import TrainingArguments
from trl import SFTTrainer

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    ArtifactType,
    WandbArtifactConfig,
    WandbArtifactLoader,
    WandbResumeMode,
    build_directory_artifact,
    default_artifact_name,
    wandb_init_from_config,
)
from lm_buddy.tasks.base import LMBuddyTask, TaskType
from lm_buddy.tasks.configs import FinetuningTaskConfig
from lm_buddy.tasks.task_output import ModelOutput, TaskOutput


def is_tracking_enabled(config: FinetuningTaskConfig) -> bool:
    """Return whether the caller is on the rank zero worker within a Ray Train context.

    Reference: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
    """
    return config.tracking and train.get_context().get_world_rank() == 0


def run_finetuning(config: FinetuningTaskConfig, artifact_loader: ArtifactLoader):
    """Run finetuning from the finetuning task config.

    This method is detached from the `FinetuningTask` so it can be executed
    on remote Ray workers without being bound to the state of the task class.
    """
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


class FinetuningTask(LMBuddyTask[FinetuningTaskConfig]):
    """Task for supervised finetuning of a causal language model."""

    def __init__(
        self,
        config: FinetuningTaskConfig,
        artifact_loader: ArtifactLoader = WandbArtifactLoader(),
    ):
        super().__init__(config)
        self._artifact_loader = artifact_loader

    @property
    def task_type(self) -> TaskType:
        return TaskType.FINETUNING

    def _run_internal(self) -> list[TaskOutput]:
        # Place the artifact loader in Ray object store
        artifact_loader_ref = ray.put(self._artifact_loader)

        # Define training function internally to capture the artifact loader ref as a closure
        # Reference: https://docs.ray.io/en/latest/ray-core/objects.html#closure-capture-of-objects
        def training_function_per_worker(config_data: dict[str, Any]):
            artifact_loader = ray.get(artifact_loader_ref)
            config = FinetuningTaskConfig(**config_data)
            if is_tracking_enabled(config):
                with wandb_init_from_config(
                    config.tracking,
                    resume=WandbResumeMode.NEVER,
                    job_type=TaskType.FINETUNING,
                ):
                    run_finetuning(config, artifact_loader)
            else:
                run_finetuning(config, artifact_loader)

        # Construct Ray train configurations from input config
        scaling_config = ScalingConfig(
            use_gpu=self.config.ray.use_gpu,
            num_workers=self.config.ray.num_workers,
        )
        run_config = RunConfig(
            name=self.config.tracking.name if self.config.tracking else None,
            storage_path=self.config.ray.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        )
        trainer = TorchTrainer(
            train_loop_per_worker=training_function_per_worker,
            train_loop_config=self.config.model_dump(),
            scaling_config=scaling_config,
            run_config=run_config,
        )
        result = trainer.fit()
        print(f"Training result: {result}")

        # Generate task outputs from training result
        task_outputs = self._get_task_outputs(result)
        return task_outputs

    def _get_task_outputs(self, result: Result) -> list[TaskOutput]:
        if result.checkpoint is None:
            # If Ray did not save a checkpoint, no outputs can be produced from the task
            return []

        ckpt_path = Path(result.checkpoint.path) / RayTrainReportCallback.CHECKPOINT_NAME
        artifact_config = self._log_model_artifact(ckpt_path)
        model_output = ModelOutput(
            path=ckpt_path,
            artifact=artifact_config,
            is_adapter=self.config.adapter is not None,
        )
        return [model_output]

    def _log_model_artifact(self, model_path: Path) -> WandbArtifactConfig | None:
        if self.config.tracking is None:
            return None

        # Register a model artifact if tracking is enabled and Ray saved a checkpoint
        with wandb_init_from_config(
            self.config.tracking,
            resume=WandbResumeMode.MUST,  # Must resume from the just-completed run
        ) as run:
            print("Logging artifact for model checkpoint...")
            artifact = build_directory_artifact(
                artifact_name=default_artifact_name(run.name, ArtifactType.MODEL),
                artifact_type=ArtifactType.MODEL,
                dir_path=model_path,
                reference=True,
            )
            self._artifact_loader.log_artifact(artifact)

            # Return reference to artifact to include in task output
            return WandbArtifactConfig(
                name=artifact.name,
                project=run.project,
                entity=run.entity,
                version="latest",
            )
