from __future__ import annotations

import tokenize
from typing import Any

from gpp.llm.qa_llm._hugging_face_agent import BaseInference, MaxTokensExceededError
from gpp.llm.qa_llm._thread import Message
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    pipeline,
)
from transformers.pipelines.base import Pipeline


class DefaultHFPipelineClient(BaseInference):
    """Default client for Hugging Face models that use the pipeline API."""

    def __init__(self, model_name: str, model_max_length: int):
        self.model_name = model_name
        self.pipe = None
        self.model_max_length = model_max_length

    def _infer(self, conversation: list[Message], max_new_tokens: int) -> str:
        pipe = self.get_pipeline()
        msgs = [msg.to_dict() for msg in conversation]
        tokenizer = pipe.tokenizer
        assert tokenizer is not None
        self.validate_input(tokenizer, msgs, max_new_tokens)
        outputs = pipe(msgs, max_new_tokens=max_new_tokens)
        return outputs[0]["generated_text"][-1]["content"].strip()  # type: ignore

    def get_pipeline(self) -> Pipeline:
        if self.pipe is None:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={},
                device_map="auto",
            )
        return self.pipe

    def validate_input(
        self, tokenizer: PreTrainedTokenizer, msgs: list[dict], max_new_tokens: int
    ):
        # make sure if the input is smaller than the model's max length + max_new_tokens
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + max_new_tokens > self.model_max_length:
            raise MaxTokensExceededError(
                f"Input length {len(tokens)} + max_new_tokens {max_new_tokens} exceeds model's max length {self.model_max_length}"
            )


class DefaultHFClient(BaseInference):
    """Default client for Hugging Face models that do not use the pipeline API."""

    def __init__(self, model_name: str, model_max_length: int):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_max_length = model_max_length

    def _infer(self, conversation: list[Message], max_new_tokens: int) -> str:
        self.load_model_and_tokenizer()
        assert self.tokenizer is not None and self.model is not None
        prompt = self.tokenizer.apply_chat_template(
            [msg.to_dict() for msg in conversation],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        if len(inputs) + max_new_tokens > self.model_max_length:
            raise MaxTokensExceededError(
                f"Input length {len(inputs)} + max_new_tokens {max_new_tokens} exceeds model's max length {self.model_max_length}"
            )

        response = self.model.generate(
            input_ids=inputs.to(self.model.device),
            max_new_tokens=max_new_tokens,
        )
        response = self.tokenizer.decode(response[0])
        return self.extract_last_response(response)

    def load_model_and_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto"
            )

    def extract_last_response(self, response: str) -> str:
        raise NotImplementedError()
