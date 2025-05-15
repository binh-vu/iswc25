from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from gp.actors.data import KGDB
from gpp.llm.qa_llm import Schema
from gpp.sem_label.isem_label import ISemLabelModel, Score, TableSemLabelAnnotation
from kgdata.models import Ontology
from libactor.actor import Actor
from libactor.cache import BackendFactory, IdentObj, cache
from libactor.misc import FnSignature, orjson_dumps
from libactor.storage._global_storage import GlobalStorage
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from sm.dataset import Example, FullTable
from sm.misc.funcs import import_attr
from sm.typing import ColumnIndex, InternalID
from smml.dataset import ColumnarDataset


@dataclass
class GppSemLabelArgs:
    model: str = field(metadata={"help": "Classpath to the Semantic Labeling Model"})
    model_args: dict = field(metadata={"help": "Arguments for the model"})
    data: str = field(metadata={"help": "Classpath to create dataset"})
    data_args: dict = field(
        metadata={"help": "Arguments for constructing the input data for the model"}
    )

    top_k_classes: int = 5
    top_k_props: int = 5

    norm: Optional[Literal["minmax"]] = None
    norm_args: Optional[tuple] = None


class GppSemLabelActor(Actor[GppSemLabelArgs]):

    VERSION = 100

    def forward(
        self,
        ex: IdentObj[Example[FullTable]],
        schema: IdentObj[Schema],
        ontology: IdentObj[Ontology],
        kgdb: IdentObj[KGDB],
    ) -> IdentObj[
        dict[
            ColumnIndex,
            tuple[list[tuple[InternalID, Score]], list[tuple[InternalID, Score]]],
        ]
    ]:
        ex_stypes = self.predict(
            IdentObj(key=ex.key, value=[ex.value]), schema, ontology, kgdb
        ).value[0]
        if self.params.norm is None:
            ex_topk_stypes = {
                ci: self.process_result(ontology.value, col_stypes)
                for ci, col_stypes in ex_stypes.items()
            }
        else:
            if self.params.norm == "minmax":
                assert self.params.norm_args is not None
                scaler = MinMaxScaler(*self.params.norm_args)
            else:
                raise ValueError(f"Unknown normalization: {self.params.norm}")

            ex_topk_stypes = {}
            for ci, col_stypes in ex_stypes.items():
                topk_classes, topk_props = self.process_result(
                    ontology.value, col_stypes
                )
                ex_topk_stypes[ci] = (
                    [
                        (l, scaler.transform(np.array(s, ndmin=2))[0, 0])
                        for l, s in topk_classes
                    ],
                    [
                        (l, scaler.transform(np.array(s, ndmin=2))[0, 0])
                        for l, s in topk_props
                    ],
                )
        return IdentObj(
            key=f"{self.key}({ex.key},{schema.key},{ontology.key})",
            value=ex_topk_stypes,
        )

    @cache(
        BackendFactory.actor.sqlite.pickle(
            mem_persist=True,
            get_dbdir=lambda self: self.get_model_dir(),
            log_serde_time="GppSemLabel.predict",
        ),
        cache_args=["examples", "schema", "ontology", "kgdb"],
        disable=os.environ.get("CACHE_GPP", "1") != "1"
        or os.environ.get("CACHE_GPP_SEM_LABEL", "1") != "1",
    )
    def predict(
        self,
        examples: IdentObj[Sequence[Example[FullTable]]],
        schema: IdentObj[Schema],
        ontology: IdentObj[Ontology],
        kgdb: IdentObj[KGDB],
        batch_size: int = 1,
        verbose: bool = False,
    ) -> IdentObj[list[TableSemLabelAnnotation]]:
        model = self.get_model()
        dataset_factory_sig, dataset_factory = self.get_data_factory()
        dataset = dataset_factory(
            **{
                argname: {
                    "examples": examples,
                    "schema": schema,
                    "ontology": ontology,
                    "kgdb": kgdb,
                }[argname]
                for argname in dataset_factory_sig.argnames
            }
        )
        predres = model.predict_dataset(dataset, batch_size=batch_size, verbose=verbose)
        # make sure that we do not have any nan -- inf is not as harmful as it's sortable
        has_nan = any(
            math.isnan(x[1])
            for expred in predres.values()
            for lst in expred.values()
            for x in lst
        )
        if has_nan:
            raise ValueError(
                "At least one of the predictions contains nan values. Double-check your results"
            )

        return IdentObj(
            key=f"{self.__class__.__name__}[{self.get_model_key()}]({examples.key},{schema.key},{ontology.key})",
            value=[predres[ex.id] for ex in examples.value],
        )

    def process_result(self, ontology: Ontology, stypes: list[tuple[str, float]]):
        """split the prediction to classes and properties"""
        cls_preds: list[tuple[InternalID, Score]] = []
        prop_preds: list[tuple[InternalID, Score]] = []

        for concept, score in stypes:
            if concept in ontology.props:
                prop_preds.append((concept, score))
            else:
                assert concept in ontology.classes
                cls_preds.append((concept, score))

        cls_preds = sorted(cls_preds, key=lambda x: x[1], reverse=True)
        prop_preds = sorted(prop_preds, key=lambda x: x[1], reverse=True)
        return (
            cls_preds[: self.params.top_k_classes],
            prop_preds[: self.params.top_k_props],
        )

    @cache(BackendFactory.actor.mem)
    def get_model(self) -> ISemLabelModel:
        cls: type[ISemLabelModel] = import_attr(self.params.model)
        return cls.load(**dict(workdir=self.get_model_dir(), **self.params.model_args))

    def get_model_dir(self) -> Path:
        model_dir = self.actor_dir.parent / "models" / self.get_model_key()
        model_dir.mkdir(exist_ok=True, parents=True)
        return model_dir

    @cache(BackendFactory.actor.mem)
    def get_model_key(self) -> str:
        return GlobalStorage.get_instance().shorten_key(
            orjson_dumps(
                {
                    "model": self.params.model,
                    "model_args": self.params.model_args,
                    "dataset": self.params.data,
                    "dataset_args": self.params.data_args,
                }
            ).decode()
        )

    @cache(BackendFactory.actor.mem)
    def get_data_factory(
        self,
    ) -> tuple[
        FnSignature,
        Callable[..., ColumnarDataset],
    ]:
        fn = import_attr(self.params.data)(**self.params.data_args)
        return FnSignature.parse(fn), fn
