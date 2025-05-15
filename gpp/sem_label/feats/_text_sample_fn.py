from __future__ import annotations

from collections import defaultdict
from typing import Sequence, TypedDict

from gpp.sem_label.feats._sample import GetSampleLabelOutput
from sm.dataset import Example, FullTable


class GetTextSampleV1OutputItem(TypedDict):
    table: dict
    context: dict
    column_index: list[int]
    column_type: list[list[str]]


def get_text_sample_v1(
    exs: Sequence[Example[FullTable]],
    samples: GetSampleLabelOutput,
) -> Sequence[GetTextSampleV1OutputItem]:
    table2index = defaultdict(list)
    for i in range(samples["sample_id"].value.shape[0]):
        table2index[samples["table_id"].value[i]].append(i)

    column_type_decoder = samples["original_column_type"].value

    output = []
    for ex in exs:
        # serialize the table into csv
        table_id = ex.table.table.table_id
        column_index = []
        column_types = []
        for i in table2index[table_id]:
            column_index.append(int(samples["column_index"].value[i]))
            column_types.append(
                [str(x) for x in column_type_decoder[samples["column_type"].value[i]]]
            )

        sample = {
            "table": ex.table.table,
            "context": ex.table.context,
            "column_index": column_index,
            "column_type": column_types,
        }
        output.append(sample)
    return output
