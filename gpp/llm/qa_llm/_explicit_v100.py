from __future__ import annotations

import re
from functools import partial
from typing import TypeAlias

from gpp.llm.qa_llm._hugging_face_agent import (
    HuggingFaceBasedAgent,
    MaxTokensExceededError,
)
from gpp.llm.qa_llm._input_table import InputTable
from gpp.llm.qa_llm._schema import Schema
from gpp.llm.qa_llm._thread import Message, Thread
from slugify import slugify

disk_slugify = partial(slugify, separator="_")
QueryResult: TypeAlias = tuple[list[tuple[int, Thread]], list[tuple[int, int, Thread]]]


class ExplicitV100(HuggingFaceBasedAgent[QueryResult]):
    name = "explicit-v100"
    system = "You are a skilled data scientist and you are asked to create a schema of the given table."
    prompt_cls = """Use the below csv table and classes to answer the question.
**Table:**\n{CSV_TABLE}\n\n**Classes:**\n{CLASSES}\n
Question: What is the best class in the list above to describe the column "{COLUMN}"?
Please keep your answer short with only the class or N/A if there is no suitable class. Also, please wrap your answer with backticks. For example, {EXAMPLE}.
    """.strip()
    prompt_prop = """Use the below csv table and properties to answer the question.
**Table:**\n{CSV_TABLE}\n\n**Properties:**\n{PROPERTIES}\n
Question: What is the best property in the list above to describe the relationship between the column "{COLUMN1}" and the column "{COLUMN2}"?
Please keep your answer short with only the property or N/A if there is no suitable property. Also, please wrap your answer with backticks. For example, {EXAMPLE}.
    """.strip()

    def query(
        self,
        table: InputTable,
        schema: Schema,
        entity_columns: list[int],
    ) -> QueryResult:
        class_content = "\n".join(
            [f"{i+1}. {s}" for i, s in enumerate(schema.class_easy_labels)]
        )
        prop_content = "\n".join(
            [f"{i+1}. {s}" for i, s in enumerate(schema.prop_easy_labels)]
        )

        name = table.id.replace("/", "_")

        csv_table = table.removed_irrelevant_table
        n_retry = 1

        cta_output = []
        for ci in entity_columns:
            for _ in range(1, 10):
                msgs = []
                if len(self.system) > 0:
                    msgs.append(Message("system", self.system))
                msgs.append(
                    Message(
                        "user",
                        self.prompt_cls.format(
                            CSV_TABLE=csv_table,
                            CLASSES=class_content,
                            COLUMN=table.index2name[ci],
                            EXAMPLE=f"`{schema.class_easy_labels[-1]}`",
                        ),
                    )
                )
                try:
                    thread = Thread(
                        f"{name}/cta_{ci}_{disk_slugify(table.index2name[ci])}",
                        msgs,
                    )
                    thread = self.chat(thread)
                    cta_output.append((ci, thread))
                    break
                except MaxTokensExceededError as e:
                    n_rows = len(table.removed_irrelevant_table_df)
                    n_rows = n_rows - 10 * n_retry
                    n_retry += 1
                    assert n_rows >= 5, "We don't want our table to be too short"
                    print(
                        f"[{table.id}] CTA prompt for column {ci} is too long. Try shorten the table to {n_rows} rows: {str(e)}"
                    )

                    csv_table = (
                        table.removed_irrelevant_table_df.head(n_rows)
                        .to_csv(index=False)
                        .strip()
                    )

        cpa_output = []
        for c1 in entity_columns:
            for c2 in table.column_indexes:
                if c1 == c2:
                    continue

                for _ in range(1, 10):
                    msgs = []
                    if len(self.system) > 0:
                        msgs.append(Message("system", self.system))
                    msgs.append(
                        Message(
                            "user",
                            self.prompt_prop.format(
                                CSV_TABLE=csv_table,
                                PROPERTIES=prop_content,
                                COLUMN1=table.index2name[c1],
                                COLUMN2=table.index2name[c2],
                                EXAMPLE=f"`{schema.prop_easy_labels[-1]}`",
                            ),
                        )
                    )
                    try:
                        thread = Thread(
                            f"{name}/cpa_{c1}_{c2}_{disk_slugify(table.index2name[c1])}__{disk_slugify(table.index2name[c2])}",
                            msgs,
                        )
                        thread = self.chat(thread)
                        cpa_output.append((c1, c2, thread))
                        break
                    except MaxTokensExceededError as e:
                        n_rows = len(table.removed_irrelevant_table_df)
                        n_rows = n_rows - 10 * n_retry
                        n_retry += 1
                        assert n_rows >= 20, "We don't want our table to be too short"
                        print(
                            f"[{table.id}] CPA prompt for columns {c1} and {c2} are too long. Try shorten the table to {n_rows} rows: {str(e)}"
                        )

                        csv_table = (
                            table.removed_irrelevant_table_df.head(n_rows)
                            .to_csv(index=False)
                            .strip()
                        )

        return (cta_output, cpa_output)

    def extract(
        self,
        table: InputTable,
        entity_columns: list[int],
        schema: Schema,
        output: QueryResult,
        can_ask_for_correction: bool = False,
    ):
        cta = {}
        cpa = []

        for ci, thread in output[0]:
            classid = self.parse_cta_answer(thread.messages[-1].content, schema)
            if classid is None or classid not in schema.class_label_keys:
                continue
            cta[ci] = schema.class_label_keys[classid]

        for c1, c2, thread in output[1]:
            propid = self.parse_cpa_answer(thread.messages[-1].content, schema)
            if propid is None or propid not in schema.prop_label_keys:
                continue
            cpa.append(
                (
                    c1,
                    c2,
                    schema.prop_label_keys[propid],
                )
            )
        return cta, cpa

    @classmethod
    def parse_cta_answer(cls, ans: str, schema: Schema):
        out = []
        for colpattern in [r'"([^"]*)"', r"`([^`]*)`"]:
            out.extend(list(re.finditer(colpattern, ans)))

        # sort by the order found in the answer
        out.sort(key=lambda x: x.span()[0])

        no_answer_keywords = ["N/A", "n/a", "N/a"]

        if any(item.group(1) in no_answer_keywords for item in out):
            return None

        classids = []
        for item in out:
            classid = cls.parse_prediction(item.group(1), True, schema)
            if classid is not None:
                classids.append(classid)
        if len(classids) == 0:
            if any(ans.find(x) != -1 for x in no_answer_keywords):
                return None

            # our parse prediction is already very forgiving -- they just hallucinate the answer
            # we can ask for correction again?
            return None
            raise ValueError(f"Cannot parse the answer: {ans}")
        return classids[0]

    @classmethod
    def parse_cpa_answer(cls, ans: str, schema: Schema):
        out = []
        for colpattern in [r'"([^"]*)"', r"`([^`]*)`"]:
            out.extend(list(re.finditer(colpattern, ans)))

        # sort by the order found in the answer
        out.sort(key=lambda x: x.span()[0])

        no_answer_keywords = ["N/A", "n/a", "N/a"]

        if any(item.group(1) in no_answer_keywords for item in out):
            return None

        propids = []
        for item in out:
            try:
                propid = cls.parse_prediction(item.group(1), False, schema)
            except:
                propid = None
            if propid is not None:
                propids.append(propid)
        if len(propids) == 0:
            if any(ans.find(x) != -1 for x in no_answer_keywords):
                return None
            return None
            # raise ValueError(f"Cannot parse the answer: {ans}")
        return propids[0]
