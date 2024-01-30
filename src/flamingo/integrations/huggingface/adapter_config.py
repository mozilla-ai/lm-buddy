import dataclasses

from peft import PeftConfig, PeftType, TaskType
from pydantic import field_validator, model_validator

from flamingo.types import BaseFlamingoConfig


def _get_peft_config_class(peft_type: PeftType) -> type[PeftConfig]:
    # Internal import to avoid bringing the global variable from peft into module scope
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]


class AdapterConfig(BaseFlamingoConfig, extra="allow"):
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

    peft_type: PeftType
    task_type: TaskType = TaskType.CAUSAL_LM

    @field_validator("peft_type", "task_type", mode="before")
    def sanitize_enum_args(cls, x):
        if isinstance(x, str):
            x = x.strip().upper()
        return x

    @model_validator(mode="after")
    def validate_adapter_args(cls, config: "AdapterConfig"):
        peft_type = config.peft_type

        # PeftConfigs are standard dataclasses so can extract their allowed field names
        adapter_cls = _get_peft_config_class(peft_type)
        allowed_fields = {x.name for x in dataclasses.fields(adapter_cls)}

        # Filter fields to those found on the PeftConfig
        extra_fields = set(config.model_fields_set).difference(allowed_fields)
        if extra_fields:
            raise ValueError(f"Unknowon arguments for {peft_type} adapter: {extra_fields}")

        return config

    def as_huggingface(self) -> PeftConfig:
        adapter_cls = _get_peft_config_class(self.peft_type)
        return adapter_cls(**self.model_dump())
