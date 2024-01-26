import dataclasses

from peft import PeftConfig, PeftType, TaskType
from pydantic import Extra, root_validator, validator

from flamingo.types import BaseFlamingoConfig


def _get_peft_config_class(peft_type: PeftType) -> type[PeftConfig]:
    # Internal import to avoid bringing the global variable from peft into module scope
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]


class AdapterConfig(BaseFlamingoConfig):
    """Configuration containing PEFT adapter settings.

    The type of adapter is controlled by the required `peft_type` field,
    which must be one of the allowed values from the PEFT `PeftType` enumeration.
    Extra arguments are allowed and are passed down to the HuggingFace `PeftConfig`
    class determined by the `peft_type` argument.

    The `task_type` for the adapter is also required.
    By default, this is set to `TaskType.CAUSAL_LM`
    which is appropriate for causal language model finetuning.
    See the allowed values in the PEFT `TaskType` enumeration.
    """

    class Config:
        extra = Extra.allow

    peft_type: PeftType
    task_type: TaskType = TaskType.CAUSAL_LM

    @validator("peft_type", "task_type", pre=True, always=True)
    def sanitize_enum_args(cls, x):
        if isinstance(x, str):
            x = x.strip().upper()
        return x

    @root_validator
    def validate_adapter_args(cls, values):
        peft_type = values["peft_type"]

        # PeftConfigs are standard dataclasses so can extract their allowed field names
        adapter_cls = _get_peft_config_class(peft_type)
        allowed_fields = {x.name for x in dataclasses.fields(adapter_cls)}

        # Filter fields to those found on the PeftConfig
        extra_fields = set(values.keys()).difference(allowed_fields)
        if extra_fields:
            raise ValueError(f"Unknowon arguments for {peft_type} adapter: {extra_fields}")

        return values

    def as_huggingface(self) -> PeftConfig:
        adapter_cls = _get_peft_config_class(self.peft_type)
        adapter_args = self.dict(exclude={"peft_type"})
        return adapter_cls(**adapter_args)
