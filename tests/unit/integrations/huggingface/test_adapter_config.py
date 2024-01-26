import pytest
from peft import PeftType, TaskType
from pydantic import ValidationError

from flamingo.integrations.huggingface import AdapterConfig
from flamingo.integrations.huggingface.adapter_config import _get_peft_config_class


def test_peft_type_sanitzation():
    config = AdapterConfig(peft_type="lora  ", task_type=TaskType.CAUSAL_LM)
    assert config.peft_type == PeftType.LORA


def test_ensure_task_type():
    with pytest.warns(UserWarning):
        config = AdapterConfig(peft_type="LORA")
        assert config.task_type == TaskType.CAUSAL_LM


def test_extra_field_validation():
    for peft_type in PeftType:
        with pytest.raises(ValidationError):
            config = AdapterConfig(
                peft_type=peft_type,
                task_type=TaskType.CAUSAL_LM,
                extra_bad_field_name="123",
            )
            assert "extra_bad_field_name" not in config.dict()


def test_huggingface_conversion():
    for peft_type in PeftType:
        config = AdapterConfig(peft_type=peft_type)

        hf_config = config.as_huggingface()
        expected_cls = _get_peft_config_class(peft_type)
        assert type(hf_config) is expected_cls
