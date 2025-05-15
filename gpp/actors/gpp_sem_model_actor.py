from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from gpp.actors.graph_space_actor import GraphSpace
from gpp.sem_label.isem_label import Score
from gpp.sem_model.from_sem_label import ISemModel
from kgdata.models import Ontology
from libactor.actor import Actor
from libactor.cache import BackendFactory, IdentObj, cache
from sm.dataset import Example, FullTable
from sm.misc.funcs import import_attr
from sm.outputs.semantic_model import SemanticModel
from sm.typing import ColumnIndex, InternalID


@dataclass
class GppSemModelArgs:
    algo: str = field(metadata={"help": "Path to the SType Algorithm"})
    algo_args: dict = field(metadata={"help": "Arguments for the algorithm"})


class GppSemModelActor(Actor[GppSemModelArgs]):
    VERSION = 100

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(mem_persist=True),
        disable=(
            os.environ.get("CACHE_GPP", "1") != "1"
            or os.environ.get("CACHE_GPP_SEM_MODEL", "1") != "1"
        ),
    )
    def forward(
        self,
        ex: IdentObj[Example[FullTable]],
        sem_label: IdentObj[
            dict[
                ColumnIndex,
                tuple[list[tuple[InternalID, Score]], list[tuple[InternalID, Score]]],
            ]
        ],
        entity_columns: Optional[IdentObj[list[ColumnIndex]]],
        graph_space: IdentObj[GraphSpace],
        ontology: IdentObj[Ontology],
    ) -> IdentObj[SemanticModel]:
        algo = self.get_algorithm()
        if entity_columns is None:
            _entity_columns = {
                ci
                for ci, (ctypes, ptypes) in sem_label.value.items()
                if len(ctypes) > 0
            }
        else:
            _entity_columns = set(entity_columns.value)

        cangraph = algo.get_candidate_graph(
            ex.value,
            sem_label.value,
            _entity_columns,
            ontology.value,
            graph_space.value,
        )
        sm = algo.get_semantic_model(ex.value, cangraph, ontology.value)
        return IdentObj(
            key=f"{self.key}({ex.key},{entity_columns.key if entity_columns else 'null'},{sem_label.key},{ontology.key},{graph_space.key})",
            value=sm,
        )

    @cache(BackendFactory.actor.mem)
    def get_algorithm(self) -> ISemModel:
        return import_attr(self.params.algo)(
            **self.params.algo_args,
        )
