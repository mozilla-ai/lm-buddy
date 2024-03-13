import contextlib
import tempfile
from pathlib import Path

from pydantic_yaml import parse_yaml_file_as, to_yaml_file

from lm_buddy.types import BaseLMBuddyConfig


class LMBuddyTaskConfig(BaseLMBuddyConfig):
    """Configuration that comprises the entire input to an LM Buddy job.

    This class implements helper methods for de/serializing the configuration from file.

    Currently, there is a 1:1 mapping between job entrypoints and job config implementations,
    but this is not rigidly constrained by the interface. This may change in the future.
    """

    @classmethod
    def from_yaml_file(cls, path: Path | str):
        return parse_yaml_file_as(cls, path)

    def to_yaml_file(self, path: Path | str):
        to_yaml_file(path, self, exclude_none=True)

    @contextlib.contextmanager
    def to_tempfile(self, *, name: str = "config.yaml", dir: str | Path | None = None):
        """Enter a context manager with the config written to a temporary YAML file.

        Keyword Args:
            name (str): Name of the config file in the tmp directory. Defaults to "config.yaml".
            dir (str | Path | None): Root path of the temporary directory. Defaults to None.

        Returns:
            Path to the temporary config file.
        """
        with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
            config_path = Path(tmpdir) / name
            self.to_yaml_file(config_path)
            yield config_path
