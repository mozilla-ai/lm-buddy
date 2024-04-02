import pytest

from lm_buddy import LMBuddy
from lm_buddy.integrations.huggingface import AutoModelConfig
from lm_buddy.integrations.wandb import WandbArtifactConfig, WandbRunConfig
from lm_buddy.jobs.configs import LMHarnessEvaluationConfig, LMHarnessJobConfig
from tests.test_utils import FakeArtifactLoader


@pytest.fixture
def job_config(llm_model_artifact):
    model_config = AutoModelConfig(
        path=WandbArtifactConfig(name=llm_model_artifact.name, project="test")
    )

    tracking_config = WandbRunConfig(name="test-lm-harness-job")
    evaluation_config = LMHarnessEvaluationConfig(tasks=["hellaswag"], limit=5)
    return LMHarnessJobConfig(
        model=model_config,
        evaluation=evaluation_config,
        tracking=tracking_config,
    )


def test_lm_harness_job_with_tracking(llm_model_artifact, job_config):
    # Preload input artifact in loader
    artifact_loader = FakeArtifactLoader()
    artifact_loader.log_artifact(llm_model_artifact)

    # Run test job
    buddy = LMBuddy(artifact_loader)
    buddy.evaluate(job_config)

    # One input artifact, and one eval artifact produced
    assert artifact_loader.num_artifacts() == 2


def test_lm_harness_job_no_tracking(llm_model_artifact, job_config):
    # Disable tracking on job config
    job_config.tracking = None

    # Preload input artifact in loader
    artifact_loader = FakeArtifactLoader()
    artifact_loader.log_artifact(llm_model_artifact)

    # Run test job
    buddy = LMBuddy(artifact_loader)
    buddy.evaluate(job_config)

    # One input artifact, no additional eval artifacts produced
    assert artifact_loader.num_artifacts() == 1
