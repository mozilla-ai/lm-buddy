import pytest

from lm_buddy import LMBuddy
from lm_buddy.integrations.huggingface import AutoModelConfig, TextDatasetConfig, TrainerConfig
from lm_buddy.integrations.wandb import ArtifactType, WandbRunConfig
from lm_buddy.jobs.configs import FinetuningJobConfig, FinetuningRayConfig
from lm_buddy.paths import format_wandb_path
from tests.utils import FakeArtifactLoader


@pytest.fixture
def job_config(llm_model_artifact, text_dataset_artifact):
    model_config = AutoModelConfig(
        path=format_wandb_path(llm_model_artifact.name, project="test"),
    )
    dataset_config = TextDatasetConfig(
        path=format_wandb_path(text_dataset_artifact.name, project="test"),
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


def test_finetuning_job(llm_model_artifact, text_dataset_artifact, job_config):
    # Preload input artifact in loader
    artifact_loader = FakeArtifactLoader()
    artifact_loader.log_artifact(llm_model_artifact)
    artifact_loader.log_artifact(text_dataset_artifact)

    # Run test job
    buddy = LMBuddy(artifact_loader)
    buddy.finetune(job_config)

    # Two input artifacts, and one output model artifact produced
    artifacts = artifact_loader.get_artifacts()
    num_dataset_artifacts = len([a for a in artifacts if a.type == ArtifactType.DATASET])
    num_model_artifacts = len([a for a in artifacts if a.type == ArtifactType.MODEL])
    assert num_dataset_artifacts == 1
    assert num_model_artifacts == 2
