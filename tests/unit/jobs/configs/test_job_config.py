from lm_buddy.jobs.configs import LMBuddyJobConfig


def test_config_as_tempfile():
    class TestConfig(LMBuddyJobConfig):
        magic_number: int

    config = TestConfig(magic_number=42)
    config_name = "my-job-config.yaml"
    with config.to_tempfile(name=config_name) as path:
        assert path.name == config_name
        assert TestConfig.from_yaml_file(path) == config
