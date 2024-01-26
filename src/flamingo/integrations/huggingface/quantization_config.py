from transformers import BitsAndBytesConfig

from flamingo.types import BaseFlamingoConfig, TorchDtypeString


class QuantizationConfig(BaseFlamingoConfig):
    """Basic quantization settings to pass to training and evaluation jobs.

    Note that in order to use BitsAndBytes quantization on Ray,
    you must ensure that the runtime environment is installed with GPU support.
    This can be configured by setting the `entrypoint_num_gpus > 0` when submitting a job
    to the cluster.
    """

    load_in_8bit: bool | None = None
    load_in_4bit: bool | None = None
    bnb_4bit_quant_type: str = "fp4"
    bnb_4bit_compute_dtype: TorchDtypeString | None = None

    def as_huggingface(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
        )
