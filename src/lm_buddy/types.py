from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator, PlainSerializer, WithJsonSchema


def validate_torch_dtype(x: Any) -> torch.dtype:
    match x:
        case torch.dtype():
            return x
        case str() if hasattr(torch, x) and isinstance(getattr(torch, x), torch.dtype):
            return getattr(torch, x)
        case _:
            raise ValueError(f"{x} is not a valid torch.dtype.")


SerializableTorchDtype = Annotated[
    torch.dtype,
    BeforeValidator(lambda x: validate_torch_dtype(x)),
    PlainSerializer(lambda x: str(x).split(".")[1]),
    WithJsonSchema({"type": "string"}, mode="validation"),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]
"""Custom type validator for a `torch.dtype` object.

Accepts `torch.dtype` instances or strings representing a valid dtype from the `torch` package.
Ref: https://docs.pydantic.dev/latest/concepts/types/#custom-types
"""


class BaseLMBuddyConfig(
    BaseModel,
    extra="forbid",
    arbitrary_types_allowed=True,
    validate_assignment=True,
):
    """Base class for all Pydantic configs in the library.

    Defines some common settings used by all subclasses.
    """
