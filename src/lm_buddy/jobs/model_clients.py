from abc import abstractmethod

import torch
from loguru import logger
from openai import OpenAI, OpenAIError
from openai.types import Completion
from transformers import pipeline

from lm_buddy.configs.common import LMBuddyConfig
from lm_buddy.configs.huggingface import AutoModelConfig
from lm_buddy.configs.jobs.hf_evaluate import HuggingFaceEvalJobConfig
from lm_buddy.configs.vllm import VLLMCompletionsConfig
from lm_buddy.jobs.asset_loader import HuggingFaceModelLoader, HuggingFaceTokenizerLoader


class BaseModelClient:
    @abstractmethod
    def __init__(self, model: str, config: LMBuddyConfig):
        pass

    @abstractmethod
    def predict(self, prompt: str) -> str:
        pass


class PipelineModelClient(BaseModelClient):
    def __init__(self, model: str, config: AutoModelConfig):
        self._summarizer = pipeline(
            "summarization",
            model=model,
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, prompt):
        # summarizer output is a list (1 element in this case) of dict with key = "summary_text"
        # TODO: bring summarizer parameters out at some point (not needed at the moment)
        pred = self._summarizer(prompt, min_length=30, do_sample=False)
        return pred[0]["summary_text"]


class HuggingFaceModelClient(BaseModelClient):
    def __init__(self, model: str, config: HuggingFaceEvalJobConfig):
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        hf_model_loader = HuggingFaceModelLoader()
        hf_tokenizer_loader = HuggingFaceTokenizerLoader()
        self._model = hf_model_loader.load_pretrained_model(config.model).to(self._device)
        self._tokenizer = hf_tokenizer_loader.load_pretrained_tokenizer(config.tokenizer)

    def predict(self, prompt):
        inputs = self._tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").to(
            self._device
        )
        generated_ids = self._model.generate(**inputs, max_new_tokens=256)
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class OpenAIModelClient(BaseModelClient):
    def __init__(self, model: str, config: VLLMCompletionsConfig):
        self._config = config

        hf_model_loader = HuggingFaceModelLoader()
        self._engine = hf_model_loader.resolve_asset_path(config.inference.engine)
        self._system = config.inference.system_prompt
        self._client = OpenAI(base_url=model)

    def _openai_chat_completion(
        self,
        config: VLLMCompletionsConfig,
        client: OpenAI,
        prompt: str,
        system: str = "You are a helpful assisant.",
    ) -> Completion:
        """Connects to a remote OpenAI-API-compatible endpoint
        and returns a chat completion holding the model's response.
        """

        return self._client.chat.completions.create(
            model=self._engine,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            frequency_penalty=config.frequency_penalty,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    def _get_response_with_retries(
        self,
        config: VLLMCompletionsConfig,
        prompt: str,
    ) -> tuple[str, str]:
        current_retry_attempt = 1
        max_retries = 1 if config.inference.max_retries is None else config.inference.max_retries
        while current_retry_attempt <= max_retries:
            try:
                response = self._openai_chat_completion(
                    self._config, self._client, prompt, self._system
                )
                break
            except OpenAIError as e:
                logger.warning(f"{e.message}: " f"Retrying ({current_retry_attempt}/{max_retries})")
                current_retry_attempt += 1
                if current_retry_attempt > max_retries:
                    raise e
        return response

    def predict(self, prompt):
        response = self._get_response_with_retries(self._config, prompt)

        return response.choices[0].message.content
