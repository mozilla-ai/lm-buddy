import contextlib
import tempfile
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Extra
from pydantic_yaml import parse_yaml_file_as, to_yaml_file


class TorchDtypeString(str):
    """String representation of a `torch.dtype`.

    Only strings corresponding to a `dtype` instance within the `torch` module are allowed.

    This class has validation and schema definitions for use in Pydantic models.
    Ref: https://docs.pydantic.dev/1.10/usage/types/#custom-data-types
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, x):
        match x:
            case torch.dtype():
                x = str(x).split(".")[1]
                return cls(x)
            case str() if hasattr(torch, x) and isinstance(getattr(torch, x), torch.dtype):
                return cls(x)
            case _:
                raise ValueError(f"{x} is not a valid torch.dtype.")

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]):
        field_schema.update(type="string", examples=["float16", "bfloat16", "int8"])

    def as_torch(self) -> torch.dtype:
        """Return the actual `torch.dtype` instance."""
        return getattr(torch, self)


class BaseFlamingoConfig(BaseModel):
    """Base class for all Pydantic configs in the library.

    Defines some common settings used by all subclasses.
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def from_yaml_file(cls, path: Path | str):
        return parse_yaml_file_as(cls, path)

    def to_yaml_file(self, path: Path | str):
        to_yaml_file(path, self, exclude_none=True)

    @contextlib.contextmanager
    def to_tempfile(self, *, name: str = "config.yaml", dir: str | Path | None = None):
        """Enter a context manager with the config written to a temporary YAML file.

        Args:
            name (str): Name of the config file in the tmp directory. Defaults to "config.yaml"
            dir (str | Path, optional): Root path of the temporary directory

        Returns:
            Path to the temporary config file
        """
        with tempfile.TemporaryDirectory(dir=dir) as tmpdir:
            config_path = Path(tmpdir) / name
            self.to_yaml_file(config_path)
            yield config_path
