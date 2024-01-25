import warnings

from peft import PeftConfig, PeftType, TaskType
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from pydantic import Extra, Field, root_validator, validator

from flamingo.types import BaseFlamingoConfig

DEFAULT_TASK_TYPE = TaskType.CAUSAL_LM


class FinetuningAdapterConfig(BaseFlamingoConfig):
    """Configuration class containing generic PEFT adapter settings.

    The type of adapter is controlled by the required `adapter_type` field,
    which must be one of the allowed values from the PEFT `PeftType` enumeration.
    Extra arguments are allowed and are passed down to the HuggingFace `PeftConfig`
    class determined by the `adapter_type` argument.
    """

    class Config:
        extra = Extra.allow

    adapter_type: PeftType = Field(..., alias="type")

    @root_validator
    def ensure_task_type(cls, values):
        if "task_type" not in values:
            warnings.warn(
                "Task type not specified for adapter. "
                f"Assuming default of `{DEFAULT_TASK_TYPE}`."
            )
            values["task_type"] = DEFAULT_TASK_TYPE
        return values

    @validator("adapter_type", pre=True, always=True)
    def sanitize_adapter_type(cls, x):
        if isinstance(x, str):
            x = x.strip().upper()
        return x

    def as_huggingface(self) -> PeftConfig:
        adapter_cls = PEFT_TYPE_TO_CONFIG_MAPPING[self.adapter_type]
        adapter_args = self.dict(exclude={"adapter_type"})
        return adapter_cls(**adapter_args)
