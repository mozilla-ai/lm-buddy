from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Extra, validator
from pydantic.fields import ModelField
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

SerializableTorchDtype = str | torch.dtype | None
"""Representation of a `torch.dtype` that can be serialized to string."""


class BaseFlamingoConfig(BaseModel):
    """Base class for all Pydantic configs in the library.

    Defines some common settings used by all subclasses.
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {
            # Default JSON encoding of a torch.dtype object
            # Defining here allows it to be inherited by all sub-classes of BaseFlamingoConfig
            torch.dtype: lambda x: str(x).split(".")[1],
        }

    @validator("*", pre=True)
    def validate_serializable_dtype(cls, x: Any, field: ModelField) -> Any:
        """Extract the torch.dtype corresponding to a string value, else return the value.

        Inspired by the HuggingFace `BitsAndBytesConfig` logic.
        """
        if field.type_ is SerializableTorchDtype and isinstance(x, str):
            return getattr(torch, x)
        return x

    @classmethod
    def from_yaml_file(cls, path: Path | str):
        return parse_yaml_file_as(cls, path)

    def to_yaml_file(self, path: Path | str):
        to_yaml_file(path, self, exclude_none=True)
