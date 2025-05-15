from __future__ import annotations

import math
from hmac import new
from pathlib import Path
from typing import Annotated, Sequence

import serde.json
from gpp.llm.qa_llm import Schema
from gpp.sem_label.isem_label import ISemLabelModel, TableSemLabelAnnotation
from kgdata.models import Ontology
from libactor.cache import IdentObj
from sm.dataset import Example, FullTable
from sm.typing import ExampleId
from smml.dataset import ColumnarDataset

DatasetId = Annotated[str, "Dataset ID"]


class PrecomputeModel(ISemLabelModel):

    def __init__(self, predict_files: list[Path]):
        data = {}
        for predict_file in predict_files:
            data[predict_file.stem] = serde.json.deser(predict_file)

        # validate the data format
        assert isinstance(data, dict)
        for ds, dsval in data.items():
            assert isinstance(ds, str) and isinstance(dsval, dict)
            for tbl, tblval in list(dsval.items()):
                assert isinstance(tbl, str)
                newtblval = {}
                if isinstance(tblval, list):
                    for item in tblval:
                        assert (
                            isinstance(item, dict)
                            and isinstance(item["column_index"], int)
                            and isinstance(item["scores"], list)
                        )
                        for score in item["scores"]:
                            assert (
                                isinstance(score, list)
                                and len(score) == 2
                                and isinstance(score[0], str)
                                and isinstance(score[1], float)
                            )
                            assert not math.isnan(score[1])
                        newtblval[item["column_index"]] = item["scores"]
                elif isinstance(tblval, dict):
                    for col, coltypes in tblval.items():
                        column_index = int(col)
                        newtblval[column_index] = coltypes
                else:
                    raise ValueError(
                        f"Invalid data format for dataset {ds}, table {tbl}: {tblval}"
                    )
                dsval[f"{ds}__{tbl}"] = newtblval
                dsval[tbl] = newtblval
        self.predictions: dict[DatasetId, dict[ExampleId, TableSemLabelAnnotation]] = (
            data
        )
        self.example_to_dataset = {}
        for ds, dspred in self.predictions.items():
            for exid in dspred.keys():
                if exid in self.example_to_dataset:
                    raise ValueError(
                        f"Example ID {exid} is duplicated in dataset {ds} and {self.example_to_dataset[exid]}"
                    )
                self.example_to_dataset[exid] = ds

    @classmethod
    def load(
        cls: type[PrecomputeModel], workdir: Path, predict_files: list[Path]
    ) -> PrecomputeModel:
        return PrecomputeModel(predict_files=predict_files)

    def predict_dataset(
        self, dataset: ColumnarDataset, batch_size: int = 8, verbose: bool = False
    ) -> dict[ExampleId, TableSemLabelAnnotation]:
        output = {}
        for i in range(len(dataset)):
            exid = dataset[i]["example_id"]
            dsid = self.example_to_dataset[exid]
            output[exid] = self.predictions[dsid][exid]
        return output


def get_dataset():
    def func(
        examples: IdentObj[Sequence[Example[FullTable]]], ontology: IdentObj[Ontology]
    ) -> ColumnarDataset:
        columns = {"table": [], "column": [], "example_id": []}
        for ex in examples.value:
            for col in ex.table.table.columns:
                columns["table"].append(ex.table)
                columns["column"].append(col.index)
                columns["example_id"].append(ex.id)
        return ColumnarDataset(columns, references={"ontology": ontology.value})

    return func
