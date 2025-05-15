from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import pandas as pd
from sm.dataset import FullTable


@dataclass
class InputTable:
    id: str
    column_names: list[str]
    column_indexes: list[int]
    name2index: dict[str, int]
    serialized_table: str
    removed_irrelevant_table: str
    removed_irrelevant_table_df: pd.DataFrame

    @cached_property
    def index2name(self):
        return dict(zip(self.column_indexes, self.column_names))

    @staticmethod
    def from_full_table(
        table: FullTable,
        sample_size: Optional[int] = 100,
        seed: Optional[int] = None,
    ) -> InputTable:
        colindex = []
        colorder = []
        colnames = []
        cols = set()
        for ci, col in enumerate(table.table.columns):
            colindex.append(col.index)
            colorder.append(ci)
            assert col.clean_name is not None

            if col.clean_name in cols:
                colnames.append(f"{col.clean_name} {col.index}")
            else:
                colnames.append(col.clean_name)
            cols.add(col.clean_name)

        df = table.table.df
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=seed)

        name2index = dict(zip(colnames, colindex))
        assert len(name2index) == len(colindex)
        # TODO: remove this -- this is just a gate to check if we can remove df[df.columns[colorder]]
        # because it is not needed
        assert len(colorder) == len(df.columns)

        return InputTable(
            id=table.table.table_id,
            column_indexes=colindex,
            column_names=colnames,
            name2index=name2index,
            serialized_table=df.to_csv(index=False).strip(),
            removed_irrelevant_table_df=df,
            removed_irrelevant_table=df.to_csv(index=False).strip(),
        )
