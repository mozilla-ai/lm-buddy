from lm_buddy import LMBuddy
from lm_buddy.jobs.configs import SimpleJobConfig
from tests.test_utils import FakeArtifactLoader


def test_simple_job():
    config = SimpleJobConfig(magic_number=42)

    buddy = LMBuddy(artifact_loader=FakeArtifactLoader())

    result = buddy.simple(config)
    assert result.magic_number == config.magic_number
