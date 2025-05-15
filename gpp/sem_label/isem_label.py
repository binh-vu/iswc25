from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional, Protocol, TypeVar

from sm.typing import ColumnIndex, ExampleId, InternalID
from smml.dataset import ColumnarDataset

C = TypeVar("C")
Score = Annotated[float, "Score of a concept from 0 - 1 (higher is better)"]
TableSemLabelAnnotation = Annotated[
    dict[ColumnIndex, list[tuple[InternalID, Score]]],
    "Semantic label annotation for a table",
]


class ISemLabelModel(Protocol):

    @classmethod
    def load(cls: type[C], workdir: Path, **kwargs) -> C: ...

    def predict_dataset(
        self, dataset: ColumnarDataset, batch_size: int = 8, verbose: bool = False
    ) -> dict[ExampleId, TableSemLabelAnnotation]: ...
