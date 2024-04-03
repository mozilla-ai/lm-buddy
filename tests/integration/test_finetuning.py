import wandb

from lm_buddy import LMBuddy
from lm_buddy.integrations.huggingface import AutoModelConfig, TextDatasetConfig, TrainerConfig
from lm_buddy.integrations.wandb import ArtifactType, WandbRunConfig
from lm_buddy.jobs.configs import FinetuningJobConfig, FinetuningRayConfig
from lm_buddy.paths import format_artifact_path
from tests.utils import FakeArtifactLoader


def get_job_config(
    model_artifact: wandb.Artifact,
    dataset_artifact: wandb.Artifact,
) -> FinetuningJobConfig:
    """Create a job config for finetuning.

    The artifacts should already be logged and contain a fully qualified W&B name.
    """
    model_config = AutoModelConfig(path=format_artifact_path(model_artifact))
    dataset_config = TextDatasetConfig(
        path=format_artifact_path(dataset_artifact),
        text_field="text",
        split="train",
    )
    trainer_config = TrainerConfig(
        max_seq_length=8,
        num_train_epochs=1,
        save_steps=1,
        save_strategy="epoch",
    )
    tracking_config = WandbRunConfig(name="test-finetuning-job")
    ray_config = FinetuningRayConfig(num_workers=1, use_gpu=False)
    return FinetuningJobConfig(
        model=model_config,
        dataset=dataset_config,
        trainer=trainer_config,
        tracking=tracking_config,
        ray=ray_config,
    )


def test_finetuning_job(llm_model_artifact, text_dataset_artifact):
    # Preload input artifact in loader
    artifact_loader = FakeArtifactLoader()
    logged_model_artifact = artifact_loader.log_artifact(llm_model_artifact)
    logged_dataset_artifact = artifact_loader.log_artifact(text_dataset_artifact)

    # Build a job config
    job_config = get_job_config(logged_model_artifact, logged_dataset_artifact)

    # Run test job
    buddy = LMBuddy(artifact_loader)
    buddy.finetune(job_config)

    # Two input artifacts, and one output model artifact produced
    artifacts = artifact_loader.get_artifacts()
    num_dataset_artifacts = len([a for a in artifacts if a.type == ArtifactType.DATASET])
    num_model_artifacts = len([a for a in artifacts if a.type == ArtifactType.MODEL])
    assert num_dataset_artifacts == 1
    assert num_model_artifacts == 2
