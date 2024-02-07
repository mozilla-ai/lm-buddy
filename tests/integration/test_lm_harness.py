import pytest

from flamingo.integrations.huggingface import AutoModelConfig
from flamingo.integrations.wandb import WandbArtifactConfig, WandbRunConfig
from flamingo.jobs.lm_harness import LMHarnessEvaluatorConfig, LMHarnessJobConfig, run_lm_harness
from tests.test_utils import FakeWandbArtifactLoader


@pytest.fixture
def job_config(gpt2_model_artifact):
    artifact_config = WandbArtifactConfig(name=gpt2_model_artifact.name, project="test")
    model_config = AutoModelConfig(load_from=artifact_config)

    tracking_config = WandbRunConfig(name="test-lm-harness-job")
    evaluator_config = LMHarnessEvaluatorConfig(tasks=["hellaswag"], limit=5)
    return LMHarnessJobConfig(
        model=model_config,
        evaluator=evaluator_config,
        tracking=tracking_config,
    )


def test_lm_harness_job_with_tracking(gpt2_model_artifact, job_config):
    # Preload input artifact in loader
    artifact_loader = FakeWandbArtifactLoader()
    artifact_loader.log_artifact(gpt2_model_artifact)

    # Run test job
    run_lm_harness(job_config, artifact_loader)

    # One input artifact, and one eval artifact produced
    assert artifact_loader.num_artifacts() == 2


def test_lm_harness_job_no_tracking(gpt2_model_artifact, job_config):
    # Disable tracking on job config
    job_config.tracking = None

    # Preload input artifact in loader
    artifact_loader = FakeWandbArtifactLoader()
    artifact_loader.log_artifact(gpt2_model_artifact)

    # Run test job
    run_lm_harness(job_config, artifact_loader)

    # One input artifact, no additional eval artifacts produced
    assert artifact_loader.num_artifacts() == 1
