from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Optional, Sequence, TypeAlias

import serde.json
from gpp.llm.qa_llm._hugging_face_agent import (
    HuggingFaceBasedAgent,
    MaxTokensExceededError,
)
from gpp.llm.qa_llm._input_table import InputTable
from gpp.llm.qa_llm._schema import InternalID, Schema
from gpp.llm.qa_llm._thread import Message, Thread
from gpp.sem_label.isem_label import ISemLabelModel, Score
from libactor.cache import IdentObj
from slugify import slugify
from sm.dataset import Example, FullTable
from sm.misc.prelude import UnreachableError
from sm.typing import ColumnIndex
from smml.dataset import ColumnarDataset
from tqdm import tqdm

disk_slugify = partial(slugify, separator="_")
QueryResult: TypeAlias = tuple[list[tuple[int, Thread]], list[tuple[int, int, Thread]]]
ExampleId = Annotated[str, "Example ID"]


class SimpleSemLabelQAModel(HuggingFaceBasedAgent[QueryResult], ISemLabelModel):
    """An LLM-based approach that does semantic labeling"""

    name = "simple-sem-label-qa"
    system = "You are a skilled data scientist and you are asked to create a schema of the given table."
    prompt = """Use the below csv table and list of types to answer the question.
**Table:**\n{CSV_TABLE}\n\n**Types:**\n{CONCEPTS}\n
Question: What is the best type in the list above to describe the column "{COLUMN}"?
Please keep your answer short with only the type or N/A if there is no suitable type. Also, please wrap your answer with backticks. For example, {EXAMPLE}.
    """.strip()

    @classmethod
    def load(cls, workdir: Path, model: str):
        return SimpleSemLabelQAModel(model=model, outdir=workdir)

    def query(
        self,
        table: InputTable,
        schema: Schema,
        entity_columns: list[int],
    ):
        raise NotImplementedError()

    def extract(
        self,
        table: InputTable,
        entity_columns: list[int],
        schema: Schema,
        output,
        can_ask_for_correction: bool = False,
    ):
        raise NotImplementedError()

    def predict_column(
        self,
        table: InputTable,
        column: int,
        schema: Schema,
    ) -> Optional[InternalID]:
        concept_content = "\n".join(
            [
                f"{i+1}. {s}"
                for i, s in enumerate(
                    schema.class_easy_labels + schema.prop_easy_labels
                )
            ]
        )
        name = slugify(table.id).replace("-", "_")

        csv_table = table.removed_irrelevant_table
        n_retry = 1

        for _ in range(1, 10):
            msgs = []
            if len(self.system) > 0:
                msgs.append(Message("system", self.system))
            msgs.append(
                Message(
                    "user",
                    self.prompt.format(
                        CSV_TABLE=csv_table,
                        CONCEPTS=concept_content,
                        COLUMN=table.index2name[column],
                        EXAMPLE=f"`{schema.prop_easy_labels[-1]}`",
                    ),
                )
            )
            try:
                thread = Thread(
                    f"{name}/col_{column}_{disk_slugify(table.index2name[column])}",
                    msgs,
                )
                thread = self.chat(thread)
                return self.parse_answer(thread.messages[-1].content, schema)
            except MaxTokensExceededError as e:
                n_rows = len(table.removed_irrelevant_table_df)
                n_rows = n_rows - 10 * n_retry
                n_retry += 1
                assert n_rows >= 20, "We don't want our table to be too short"
                print(
                    f"[{table.id}] prompt for column {column} is too long. Try shorten the table to {n_rows} rows: {str(e)}"
                )
                csv_table = (
                    table.removed_irrelevant_table_df.head(n_rows)
                    .to_csv(index=False)
                    .strip()
                )

        raise UnreachableError("We should have returned in the loop")

    def parse_answer(self, ans: str, schema: Schema) -> Optional[InternalID]:
        out = []
        for colpattern in [r'"([^"]*)"', r"`([^`]*)`"]:
            out.extend(list(re.finditer(colpattern, ans)))

        # sort by the order found in the answer
        out.sort(key=lambda x: x.span()[0])

        no_answer_keywords = ["N/A", "n/a", "N/a"]

        if any(item.group(1) in no_answer_keywords for item in out):
            return None

        class_or_prop_ids = []
        for item in out:
            class_or_prop_id = self.parse_prediction(item.group(1), True, schema)
            if class_or_prop_id is None:
                class_or_prop_id = self.parse_prediction(item.group(1), False, schema)

            if class_or_prop_id in schema.class_label_keys:
                class_or_prop_id = schema.class_label_keys[class_or_prop_id]
            elif class_or_prop_id in schema.prop_label_keys:
                class_or_prop_id = schema.prop_label_keys[class_or_prop_id]

            if class_or_prop_id is not None:
                class_or_prop_ids.append(class_or_prop_id)

        if len(class_or_prop_ids) == 0:
            if any(ans.find(x) != -1 for x in no_answer_keywords):
                return None

            # our parse prediction is already very forgiving -- they just hallucinate the answer
            # we can ask for correction again?
            return None
            raise ValueError(f"Cannot parse the answer: {ans}")
        return class_or_prop_ids[0]

    def predict_dataset(
        self, dataset: ColumnarDataset, batch_size: int = 8, verbose: bool = False
    ) -> dict[ExampleId, dict[ColumnIndex, list[tuple[InternalID, Score]]]]:
        schema = dataset.references["schema"].value
        output = defaultdict(dict)
        for i in tqdm(range(len(dataset)), "Semantic Labeling"):
            pred = self.predict_column(
                dataset[i]["table"], dataset[i]["column"], schema
            )
            if pred is None:
                continue
            ex_id = dataset[i]["example_id"]
            output[ex_id][dataset[i]["column"]] = [(pred, 0.5)]
        return output


def get_dataset(
    sample_size: Annotated[int, "Sample the input table to a fixed size"],
    seed: Annotated[int, "seed for random generator"],
):
    def func(
        examples: IdentObj[Sequence[Example[FullTable]]], schema: IdentObj[Schema]
    ) -> ColumnarDataset:
        columns = {"table": [], "column": [], "example_id": []}
        for ex in examples.value:
            table = InputTable.from_full_table(ex.table, sample_size, seed)
            for ci in table.column_indexes:
                columns["table"].append(table)
                columns["column"].append(ci)
                columns["example_id"].append(ex.id)
        return ColumnarDataset(columns, references={"schema": schema})

    return func
