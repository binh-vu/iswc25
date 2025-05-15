from __future__ import annotations

from collections import defaultdict
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence, TypeVar
from uuid import uuid4

import pandas as pd
import serde.json
from gp.actors.data import KGDB
from gpp.sem_label.data_modules.v210 import SLabelV210DataModule
from gpp.sem_label.feats import (
    GetExamplesArgs,
    get_class_distance,
    get_examples,
    get_property_distance,
    get_sample_label,
    get_target_label,
    get_target_label_examples,
    get_text_embedding,
    get_text_sample_v1,
)
from gpp.sem_label.isem_label import ISemLabelModel, TableSemLabelAnnotation
from gpp.sem_label.models.contrastive.v220 import V220
from libactor.cache import IdentObj
from sm.dataset import Example, FullTable
from sm.misc.funcs import import_attr
from sm.typing import ExampleId
from smml.dataset import ColumnarDataset
from torch.utils.data import DataLoader


class SemLabelV220(ISemLabelModel):

    def __init__(
        self,
        model: V220,
        # datamodule: SLabelV210DataModule,
        # params: dict,
    ) -> None:
        self.model = model
        # self.datamodule = datamodule
        # self.params = params

    @classmethod
    def load(
        cls: type[SemLabelV220],
        workdir: Path,
        ckpt_file: Path,
        # model_params: Path
        clspath: str = "gpp.sem_label.models.contrastive.v220.V220",
    ) -> SemLabelV220:
        # params = serde.json.deser(model_params)
        # datamodule = SLabelV210DataModule(
        #     workdir,
        #     params["embedding_model"],
        #     get_text_embedding(params["embedding_model"]),
        #     train_batch_size=32,
        #     eval_batch_size=32,
        #     is_cta_only=params["is_cta_only"],
        # )
        modelcls: V220 = import_attr(clspath)
        model = modelcls.load_from_checkpoint(ckpt_file)

        return SemLabelV220(
            model=model.eval(),
            # datamodule=datamodule,
            # params=params,
        )

    def predict_dataset(
        self,
        dataset: ColumnarDataset,
        batch_size: int = 8,
        verbose: bool = False,
        evaluate: bool = False,
    ) -> dict[ExampleId, TableSemLabelAnnotation]:
        self.model.set_references(dataset)
        if evaluate:
            func = self.model.evaluate
        else:
            func = self.model.predict
        predout = func(
            DataLoader(dataset, batch_size=batch_size, pin_memory=True), verbose
        )

        preds = predout["preds"]
        targets = predout["targets"]
        sample_id = predout["indexes"]

        output = defaultdict(dict)

        target_labels = dataset.references["label_id"]

        for i, sidx in enumerate(sample_id):
            table_id = dataset.references["table_id"][sidx[0]]
            column_index = int(dataset.references["col_index"][sidx[0]])
            # assert record["sample_id"] == sidx
            output[table_id][column_index] = sorted(
                zip(target_labels, preds[i].tolist()), key=itemgetter(1), reverse=True
            )

        return dict(output)


def make_raw_dataset(
    example_dir: Path,
    exs: Sequence[Example[FullTable]],
    kgdb: IdentObj[KGDB],
    ignore_no_type_column: bool = True,
    target_label_ids: Optional[Sequence[str]] = None,
    n_examples_per_label: int = 100,
):
    cls_distance = get_class_distance(kgdb.value.ontology).value
    prop_distance = get_property_distance(kgdb.value.ontology).value

    ontology = kgdb.value.ontology.value

    wrapped_get_examples = partial(
        get_examples,
        entdb=RedirectMapping(
            kgdb.value.pydb.entity_redirections,
            kgdb.value.pydb.entity_metadata.cache(),
        ),
        ontology=ontology,
        args=GetExamplesArgs(manual_example_dir=example_dir),
    )

    samples = get_sample_label(exs, ontology.kgns, ignore_no_type_column)
    text_samples = get_text_sample_v1(exs, samples)
    target_labels = get_target_label(
        samples, ontology, cls_distance, prop_distance, target_label_ids
    )
    target_label_examples = get_target_label_examples(
        target_labels,
        wrapped_get_examples([l["id"] for l in target_labels]),
        n_examples_per_label,
    )

    return {
        "samples": text_samples,
        "target_labels": pd.DataFrame(data=target_labels).set_index("id"),
        "target_label_examples": {x["id"]: x["example"] for x in target_label_examples},
    }


def get_dataset(
    example_dir: Path,
    model_params: Path,
    n_examples_per_column: int = 100,
    n_examples_per_label: int = 100,
    ignore_no_type_column: bool = True,
    target_label_ids: Optional[Sequence[str]] = None,
):
    params = serde.json.deser(model_params)
    datamodule = SLabelV210DataModule(
        f"/tmp/{uuid4()}",  # we do not need data_dir in this case.
        params["embedding_model"],
        get_text_embedding(params["embedding_model"]),
        train_batch_size=32,
        eval_batch_size=32,
        n_examples_per_column=n_examples_per_column,
        n_examples_per_label=n_examples_per_label,
    )

    def func(
        examples: IdentObj[Sequence[Example[FullTable]]],
        kgdb: IdentObj[KGDB],
    ) -> ColumnarDataset:
        name = str(uuid4())
        is_train = False
        dataset = make_raw_dataset(
            example_dir,
            examples.value,
            kgdb,
            ignore_no_type_column,
            target_label_ids,
            n_examples_per_label,
        )
        dataset = datamodule.transformation(name, dataset)
        dataset = datamodule.make_columnar_dataset(
            name, dataset, embedding_readonly=False
        )
        datamodule.embedding_manager.flush(soft=False)
        return dataset

    return func


K = TypeVar("K")
V = TypeVar("V")


class RedirectMapping(Mapping[K, V]):
    def __init__(self, mapping1: Mapping[K, K], mapping2: Mapping[K, V]) -> None:
        self.mapping1 = mapping1
        self.mapping2 = mapping2

    def __getitem__(self, key: K) -> V:
        if key in self.mapping1:
            return self.mapping2[self.mapping1[key]]
        return self.mapping2[key]

    def __iter__(self) -> Iterator[K]:
        raise NotImplementedError()

    def values(self) -> Iterator[V]:
        return self.mapping2.values()  # type: ignore

    def __len__(self):
        return len(self.mapping2)

    def __contains__(self, key):
        if key in self.mapping1:
            return True
        return key in self.mapping2

    def get(self, key: K, default=None):
        if key in self.mapping1:
            return self.mapping2[self.mapping1[key]]
        return self.mapping2.get(key, default)
