from __future__ import annotations

from gpp.llm.qa_llm._hfmodels._default import DefaultHFClient, DefaultHFPipelineClient
from gpp.llm.qa_llm._thread import Message


class Gemma2Pipeline(DefaultHFPipelineClient):
    """Gemma 2 does not support system role.

    See https://github.com/abetlen/llama-cpp-python/issues/1580
    """

    def _infer(self, conversation: list[Message], max_new_tokens: int) -> str:
        if len(conversation) > 0 and conversation[0].role == "system":
            new_msgs = conversation[1:]
            assert new_msgs[0].role == "user"
            new_msgs[0].content = conversation[0].content + "\n" + new_msgs[0].content
        else:
            new_msgs = conversation

        return super().infer(
            new_msgs,
            max_new_tokens,
        )


class Gemma2(DefaultHFClient):
    """Gemma 2 does not support system role.

    See https://github.com/abetlen/llama-cpp-python/issues/1580

    I encounter an error while using DefaultHFPipelineClient that variable is not aligned - use DefaultHFClient instead
    """

    def _infer(self, conversation: list[Message], max_new_tokens: int) -> str:
        if len(conversation) > 0 and conversation[0].role == "system":
            new_msgs = conversation[1:]
            assert new_msgs[0].role == "user"
            new_msgs[0].content = conversation[0].content + "\n" + new_msgs[0].content
        else:
            new_msgs = conversation

        return super().infer(
            new_msgs,
            max_new_tokens,
        )

    def extract_last_response(self, response: str) -> str:
        user, assistant = response.split("<start_of_turn>model")
        return assistant.split("<end_of_turn>")[0].strip()
