from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import orjson


@dataclass
class Message:
    role: Literal["user", "system", "assistant"]
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


@dataclass
class Thread:
    """A thread is a data structure to hold conversation between a user and an LLM assistant. It is a list of messages."""

    name: str
    messages: list[Message]

    def user_reply(self, content: str) -> Thread:
        return Thread(
            name=self.name,
            messages=self.messages + [Message(role="user", content=content)],
        )

    def assistant_reply(self, content: str) -> Thread:
        return Thread(
            name=self.name,
            messages=self.messages + [Message(role="assistant", content=content)],
        )

    @staticmethod
    def load(dir: Path) -> Thread:
        obj = orjson.loads((dir / "data.json").read_bytes())
        updated_thread = Thread(
            name=obj["name"],
            messages=[Message(**msg) for msg in obj["messages"]],
        )
        return updated_thread

    def save(self, outdir: Path):
        """Save the thread to a directory (assuming that the directory exists). We have a thread into two files:

        1. conversation.txt - the conversation in a human-readable format
        2. data.json - the thread in JSON
        """
        (outdir / "data.json").write_bytes(
            orjson.dumps(
                {
                    "name": self.name,
                    "messages": [asdict(msg) for msg in self.messages],
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        conversations = []
        for msg in self.messages:
            conversations.append("-" * 4 + f" {msg.role: <10} " + "-" * 80)
            conversations.append(msg.content)
        (outdir / "conversation.txt").write_text("\n".join(conversations))
        (outdir / "_SUCCESS").touch()

    @staticmethod
    def does_save_dir_exist(outdir: Path) -> bool:
        return outdir.exists() and (outdir / "_SUCCESS").exists()

    def hashed_content(self) -> str:
        content = orjson.dumps(
            {
                "messages": [asdict(msg) for msg in self.messages],
            }
        )
        return sha256(content).hexdigest()
