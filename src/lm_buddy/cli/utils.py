from pathlib import Path
from typing import TypeVar

from lm_buddy.jobs.configs.base import LMBuddyJobConfig

ConfigType = TypeVar("ConfigType", bound=LMBuddyJobConfig)


def parse_config_option(config_cls: type[ConfigType], config: str) -> ConfigType:
    if Path(config).exists():
        return config_cls.from_yaml_file(config)
    else:
        return config_cls.model_validate_json(config)
