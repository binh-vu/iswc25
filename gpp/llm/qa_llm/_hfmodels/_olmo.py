from __future__ import annotations

from gpp.llm.qa_llm._hugging_face_agent import BaseInference
from gpp.llm.qa_llm._thread import Message
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFOLMoClient(BaseInference):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

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
        response = self.model.generate(
            input_ids=inputs.to(self.model.device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        response = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        user, assistant = response.split("<|assistant|>")
        return assistant.strip()

    def load_model_and_tokenizer(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
            except ImportError:
                print(
                    "If it requires to install `hf_olmo` package, please install `ai2-olmo` instead. See https://huggingface.co/allenai/OLMo-7B/discussions/3 for more."
                )
                raise
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto"
            )


if __name__ == "__main__":
    print(
        HFOLMoClient("allenai/OLMo-2-1124-7B-Instruct").infer(
            [Message(role="user", content="What is the capital of France?")],
            max_new_tokens=200,
        )
    )
