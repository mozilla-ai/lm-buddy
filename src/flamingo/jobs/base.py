import contextlib
import tempfile
from pathlib import Path

from flamingo.types import BaseFlamingoConfig


class BaseJobConfig(BaseFlamingoConfig):
    """A `BaseFlamingoConfig` with some additional functionality to support job entrypoints.

    Currently, there is a 1:1 mapping between job config implementations and job entrypoints.
    We may want to tighten these interfaces in the future to make implementing that
    relationship more of a mandatory feature of the library.
    """

    @contextlib.contextmanager
    def to_tempfile(self, *, name: str | None = None, dir: str | Path | None = None):
        """Enter a context manager with the config written to a temporary YAML file.

        Args:
            name (str, optional): Name of the config file in the temporary directory
            dir (str | Path, optional): Root path of the temporary directory

        Returns:
            Path to the temporary config file
        """
        config_name = name or "config.yaml"
        with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
            config_path = Path(tmpdir) / config_name
            self.to_yaml_file(config_path)
            yield config_path
