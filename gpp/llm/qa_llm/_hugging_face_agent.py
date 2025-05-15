from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from gpp.llm.qa_llm._base_agent import BaseAgent, R
from gpp.llm.qa_llm._schema import Schema
from gpp.llm.qa_llm._thread import Message, Thread
from rdflib import logger
from sm.misc.ray_helper import RemoteClient


class MaxTokensExceededError(Exception):
    pass


class BaseInference(ABC):
    @staticmethod
    def from_model_name(model: str) -> BaseInference:
        from gpp.llm.qa_llm._hfmodels._default import DefaultHFPipelineClient
        from gpp.llm.qa_llm._hfmodels._gemma_2 import Gemma2, Gemma2Pipeline
        from gpp.llm.qa_llm._hfmodels._olmo import HFOLMoClient

        if model.startswith("allenai/OLMo"):
            # e.g. "allenai/OLMo-7B-Instruct", "allenai/OLMo-2-1124-7B-Instruct"
            cls = HFOLMoClient
            args = (model,)
        elif model.startswith("google/gemma-2-"):
            # cls = Gemma2
            cls = Gemma2Pipeline
            max_context_length = 8192
            args = (model, max_context_length)
        else:
            if model.startswith("meta-llama/Llama-2"):
                # e.g., meta-llama/Llama-2-70b-chat-hf, meta-llama/Llama-2-7b-chat-hf
                max_context_length = 4096
            elif model.startswith("meta-llama/Meta-Llama-3.1"):
                max_context_length = 128000
            else:
                raise NotImplementedError(f"Model {model} is not implemented yet")

            cls = DefaultHFPipelineClient
            args = (model, max_context_length)

        if os.environ.get("HF_REMOTE") is not None:
            logger.info(
                "Using a remote server to run HuggingFace model. The server must be running and accessible through localhost port forwarding"
            )
            client = RemoteClient(cls, (model,), os.environ["HF_REMOTE"])
            return cast(BaseInference, client)

        return cls(*args)

    def infer(self, conversation: list[Message], max_new_tokens: int) -> str:
        try:
            return self._infer(conversation, max_new_tokens)
        except Exception as e:
            if str(e).startswith(
                "Input validation error: `inputs` must have less than"
            ):
                raise MaxTokensExceededError() from e
            raise

    @abstractmethod
    def _infer(self, conversation: list[Message], max_new_tokens: int) -> str: ...


class HuggingFaceBasedAgent(BaseAgent[R]):
    name: str
    system: str
    prompt: str

    def __init__(self, model: str, outdir: Path, max_new_tokens: int = 100):
        self.outdir = outdir
        self.client = BaseInference.from_model_name(model)
        self.max_new_tokens = max_new_tokens

    def chat(self, thread: Thread) -> Thread:
        thread_dir: Path = self.get_thread_dir(thread)
        if Thread.does_save_dir_exist(thread_dir):
            updated_thread = Thread.load(thread_dir)
            assert (
                thread.name == updated_thread.name
                and thread.messages == updated_thread.messages[:-1]
            )
            return updated_thread

        response = self.client.infer(thread.messages, self.max_new_tokens)
        updated_thread = thread.assistant_reply(response)
        updated_thread.save(self.get_thread_dir(thread))
        return updated_thread

    def get_thread_dir(self, thread: Thread) -> Path:
        # we don't run slugify to the name because we want to allow subdirectories if they want.
        outdir = self.outdir / self.name / "threads" / thread.name
        outdir = outdir / f"{len(thread.messages):03d}__{thread.hashed_content()[:7]}"
        outdir.mkdir(exist_ok=True, parents=True)
        return outdir

    @classmethod
    def parse_prediction(cls, val: str, is_class: bool, schema: Schema):
        # try to find by id first.
        lst = re.findall(r"((?:Q|P)\d+)", val)
        if len(lst) > 0:
            # assert len(lst) == 1
            return lst[0]

        try:
            if is_class:
                return schema.classes[schema.class_labels.index(val)]
            else:
                return schema.props[schema.prop_labels.index(val)]
        except ValueError:
            return None
