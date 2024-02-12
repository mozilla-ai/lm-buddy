import ray

from lm_buddy.jobs.simple import SimpleJobConfig
from lm_buddy.jobs.simple.entrypoint import get_magic_number


def test_simple_remote_task():
    config = SimpleJobConfig(magic_number=42)
    result = ray.get(get_magic_number.remote(config))
    assert result == 42
