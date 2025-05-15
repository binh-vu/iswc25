from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Generic, TypeVar

from gpp.llm.qa_llm._input_table import InputTable
from gpp.llm.qa_llm._schema import Schema

R = TypeVar("R")


SourceColumn = Annotated[int, "Source Column Index"]
TargetColumn = Annotated[int, "Target Column Index"]
PropertyId = Annotated[str, "Property ID (e.g., P131)"]
ClassId = Annotated[str, "Class ID (e.g., Q5)"]

CPA = Annotated[
    list[tuple[SourceColumn, TargetColumn, PropertyId]], "Column Property Assignment"
]
CTA = Annotated[dict[SourceColumn, ClassId], "Column Type Assignment"]


class BaseAgent(Generic[R], ABC):

    @abstractmethod
    def query(
        self,
        table: InputTable,
        schema: Schema,
        entity_columns: list[int],
    ) -> R: ...

    @abstractmethod
    def extract(
        self,
        table: InputTable,
        entity_columns: list[int],
        schema: Schema,
        output: R,
        can_ask_for_correction: bool = False,
    ) -> tuple[CTA, CPA]: ...
