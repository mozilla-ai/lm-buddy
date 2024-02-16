from lm_buddy.jobs.configs import SimpleJobConfig


def test_config_as_tempfile():
    config = SimpleJobConfig(magic_number=42)
    config_name = "my-job-config.yaml"
    with config.to_tempfile(name=config_name) as path:
        assert path.name == config_name
        assert SimpleJobConfig.from_yaml_file(path) == config
