from lm_buddy.jobs.configs import LMBuddyJobConfig
from lm_buddy.paths import AssetPath


def test_config_as_tempfile():
    class TestConfig(LMBuddyJobConfig):
        def asset_paths(self) -> set[AssetPath]:
            return super().asset_paths()

    config = TestConfig(name="test-config")
    config_name = "my-job-config.yaml"
    with config.to_tempfile(name=config_name) as path:
        assert path.name == config_name
        assert TestConfig.from_yaml_file(path) == config
