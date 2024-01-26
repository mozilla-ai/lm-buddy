import pytest
from peft import PeftType, TaskType
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

from flamingo.integrations.huggingface import AdapterConfig


def test_ensure_task_type():
    with pytest.warns(UserWarning):
        config = AdapterConfig(adapter_type="LORA")
        assert config.task_type == TaskType.CAUSAL_LM


def test_adapter_type_sanitzation():
    config = AdapterConfig(adapter_type="lora  ", task_type=TaskType.CAUSAL_LM)
    assert config.adapter_type == PeftType.LORA


def test_huggingface_conversion():
    for adapter_type in PeftType:
        config = AdapterConfig(adapter_type=adapter_type)

        hf_config = config.as_huggingface()
        expected_cls = PEFT_TYPE_TO_CONFIG_MAPPING[adapter_type]
        assert type(hf_config) is expected_cls
