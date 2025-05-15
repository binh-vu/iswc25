from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpp.llm.qa_llm import BaseAgent, InputTable, Schema
from libactor.actor import Actor
from libactor.cache import IdentObj, fmt_keys
from loguru import logger
from sm.dataset import FullTable
from sm.misc.funcs import import_attr
from sm.outputs import SemanticModel, create_sm_from_cta_cpa, create_sm_nodes


@dataclass
class QALLMActorArgs:
    model: str = field(metadata={"help": "Path to the LLM Model"})
    model_args: dict = field(metadata={"help": "Arguments for the model"})
    sample_size: Optional[int] = field(
        metadata={"help": "Sample the input table to a fixed size"}
    )
    seed: Optional[int] = field(metadata={"help": "Seed for random generator"})
    can_ask_for_correction: bool = field(
        default=True, metadata={"help": "Correct itself if needed"}
    )


class QALLMActor(Actor[QALLMActorArgs]):
    """LLM that handle semantic modeling using Q/A approach"""

    VERSION = 100

    def __init__(self, params: QALLMActorArgs):
        super().__init__(params)
        self.agent: Optional[BaseAgent] = None

    def forward(
        self,
        input: IdentObj[FullTable],
        entity_columns: IdentObj[list[int]],
        schema: IdentObj[Schema],
    ) -> IdentObj[SemanticModel]:
        agent = self.get_agent()
        table = InputTable.from_full_table(
            input.value, self.params.sample_size, self.params.seed
        )
        ans = agent.query(table, schema.value, entity_columns.value)

        try:
            cta, cpa = agent.extract(
                table,
                entity_columns.value,
                schema.value,
                ans,
                can_ask_for_correction=self.params.can_ask_for_correction,
            )
        except:
            print(f"Error in extracting from table: {input.value.table.table_id}")
            raise

        sm = create_sm_from_cta_cpa(
            kgns=schema.value.kgns,
            nodes=create_sm_nodes(
                {
                    ci: input.value.table.get_column_by_index(ci).clean_name or ""
                    for ci in table.column_indexes
                }
            ),
            cta=cta,
            cpa=cpa,
            get_cls_label=schema.value.get_class_label,
            get_prop_label=schema.value.get_prop_label,
        )

        key = (
            input.key
            + "|"
            + fmt_keys(
                actor=self.key,
                schema=schema,
                entity_columns=entity_columns,
            )
        )

        return IdentObj(key=key, value=sm)

    def get_agent(self) -> BaseAgent:
        if self.agent is None:
            logger.debug("Working directory for agent: {}", self.actor_dir)
            args = dict(self.params.model_args, outdir=self.actor_dir)
            self.agent = import_attr(self.params.model)(**args)
        return self.agent  # type: ignore
