from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import lightning as L
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as FM
from loguru import logger
from sm.misc.funcs import assert_isinstance
from smml.dataset import ColumnarDataset, EmbeddingFeat
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from tqdm.auto import tqdm


@dataclass
class ModelOutput:
    # only have valid values if labeled data is provided
    loss: torch.Tensor
    # scores of the predictions -- higher is better
    scores: torch.Tensor


@dataclass
class DatasetRef:
    refs: tuple[torch.Tensor, ...]
    device: Optional[str]


class BaseLightningModule(pl.LightningModule):
    def __init__(self, val_dataloader_names: list[str], metrics: list[str]):
        super().__init__()
        self.hits = []
        self.hits_at_k = [1, 3, 5, 10, 20]
        self.metrics = metrics

        if len(val_dataloader_names) > 0:
            self.valsets = [f"val_{name}" for name in val_dataloader_names]
        else:
            self.valsets = ["val"]

        # make metrics first
        subsets = ["train", "test"] + self.valsets
        for subset in subsets:
            for metric in metrics:
                if metric == "hit":
                    for k in self.hits_at_k:
                        self.get_metric(subset, f"hit{k}")
                else:
                    self.get_metric(subset, metric)

    def training_step(self, batch, batch_idx):
        output = self._pl_step(batch, batch_idx, "train")
        return output.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._pl_step(batch, batch_idx, self.valsets[dataloader_idx])

    def test_step(self, batch, batch_idx):
        self._pl_step(batch, batch_idx, "test")

    def get_metric(self, subset: str, metric: str):
        attr = f"{subset}_{metric}"
        if not hasattr(self, attr):
            if metric == "mrr":
                setattr(self, attr, RetrievalMRR(empty_target_action="error"))
            elif metric.startswith("hit"):
                topk = int(metric[3:])
                setattr(
                    self,
                    attr,
                    RetrievalHitRate(top_k=topk, empty_target_action="error"),
                )
            elif metric == "binary_precision":
                setattr(self, attr, BinaryPrecision())
            elif metric == "binary_recall":
                setattr(self, attr, BinaryRecall())
            else:
                raise NotImplementedError()
        return getattr(self, attr)

    def _pl_step(self, batch, batch_idx, prefix):
        raise NotImplementedError()

    def evaluate(self, dl: DataLoader):
        trainer = pl.Trainer(accelerator="gpu", devices=1)
        evalout = trainer.test(self, dataloaders=dl)[0]
        evalout = {
            key: value for key, value in evalout.items() if key.find("loss") == -1
        }

        keys = []
        values = []
        for metric in self.metrics:
            if metric == "hit":
                for k in self.hits_at_k:
                    try:
                        (value,) = [
                            val
                            for key, val in evalout.items()
                            if key.endswith(f"_hit{k}")
                        ]
                    except:
                        raise KeyError(f"hit{k} not found in {evalout.keys()}")
                    keys.append(f"hit{k}")
                    values.append(value)
            else:
                try:
                    (value,) = [
                        val
                        for key, val in evalout.items()
                        if key.endswith("_" + metric)
                    ]
                except:
                    raise KeyError(f"{metric} not found in {evalout.keys()}")
                keys.append(metric)
                values.append(value)

        logger.info(
            "for copying...\ndataset\t{}\n{}",
            "\t".join(keys),
            ",".join(
                [
                    assert_isinstance(dl.dataset, ColumnarDataset).name,
                ]
                + ["%.2f" % (round(float(x) * 100, 2)) for x in values]
            ),
        )


class V210BaseContrastiveLearningModel(BaseLightningModule):
    EXPECTED_EVAL_ARGS: set[str] = set()

    def __init__(self, val_dataloader_names: list[str], metrics: list[str]):
        super().__init__(val_dataloader_names, metrics)
        self.dataset2references: dict[int, DatasetRef] = {}

    def set_references(
        self,
        dataset: ColumnarDataset | L.LightningDataModule,
    ):
        if isinstance(dataset, ColumnarDataset):
            dataset_id = dataset.columns["dataset_id"][0]
            logger.info("set references for dataset {} ({})", dataset.name, dataset_id)
            self.dataset2references[dataset_id] = DatasetRef(
                refs=(
                    torch.from_numpy(dataset.references["label_name"].to_array()).to(
                        self.device
                    ),
                    torch.from_numpy(dataset.references["label_desc"].to_array()).to(
                        self.device
                    ),
                    torch.from_numpy(dataset.references["label_example"].to_array()).to(
                        self.device
                    ),
                ),
                device=None,
            )
        else:
            assert isinstance(dataset, L.LightningDataModule)
            for key in ["train", "dev", "devs", "val", "vals", "test", "tests"]:
                if not hasattr(dataset, key):
                    continue
                attr = getattr(dataset, key)
                if attr is None:
                    continue
                if isinstance(attr, ColumnarDataset):
                    self.set_references(attr)
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, ColumnarDataset):
                            self.set_references(item)
                        elif isinstance(item, list):
                            for subitem in item:
                                if isinstance(subitem, ColumnarDataset):
                                    self.set_references(subitem)

    def get_label_input(self, dataset_id: int):
        ref = self.dataset2references[dataset_id]
        if ref.device is None:
            ref.refs = tuple(ref_.to(self.device) for ref_ in ref.refs)
        return ref.refs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

    def _pl_step(self, batch, batch_idx, prefix):
        output = self.forward(**{arg: batch[arg] for arg in self.EXPECTED_ARGS})
        # shape of the output is B x C --- where B is the batch size, C is the number of classes
        batch_size, num_classes = output.scores.shape

        indexes = (
            batch["indexes"]
            .view(batch_size, 1)
            .expand(batch_size, num_classes)
            .reshape(-1)
        )
        preds = output.scores.view(-1)
        targets = batch["column_targets"].view(-1)

        if prefix == "train":
            mask = batch["column_targets_mask"].view(-1)
            assert mask.dtype is torch.bool
            indexes = indexes[mask]
            preds = preds[mask]
            targets = targets[mask]
        else:
            assert prefix == "test" or prefix.startswith("val_"), prefix

        self.log(
            f"{prefix}_loss",
            output.loss,
            prog_bar=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

        metric_name = f"{prefix}_mrr"
        metric = self.get_metric(prefix, "mrr")
        metric(preds, targets, indexes=indexes)

        self.log(
            metric_name,
            metric,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            metric_attribute=metric_name,
            add_dataloader_idx=False,
        )
        for kidx, k in enumerate(self.hits_at_k):
            metric_name = f"{prefix}_hit{k}"
            metric = self.get_metric(prefix, f"hit{k}")
            metric(preds, targets, indexes=indexes)
            self.log(
                metric_name,
                metric,
                prog_bar=kidx < 3,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
                metric_attribute=metric_name,
                add_dataloader_idx=False,
            )
        return output

    def predict(self, dataloader: DataLoader, verbose: bool = False):
        self.eval()

        assert isinstance(dataloader.dataset, ColumnarDataset)
        references = dataloader.dataset.references

        indexes = []
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval", disable=not verbose):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch = self.transform_eval_batch(references, batch)

                # after transform eval input, we must have "targets" containing 0 - 1 labels
                # of the correct classes and "indexes" containing the id of each example
                assert "column_targets" in batch and "indexes" in batch

                output = self.forward(
                    **{arg: batch[arg] for arg in self.EXPECTED_EVAL_ARGS}
                )

                preds.append(output.scores)
                targets.append(batch["column_targets"])
                indexes.append(batch["indexes"])

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        indexes = torch.cat(indexes, dim=0)
        return {"preds": preds, "targets": targets, "indexes": indexes}

    def evaluate(
        self,
        dataloader: DataLoader,
        version: int | str = "-",
    ):
        """Evaluate the performance of the model on the given dataset."""
        predres = self.predict(dataloader)

        preds = predres["preds"]
        targets = predres["targets"]
        indexes = predres["indexes"]

        out = {
            "result": {
                "mrr": RetrievalMRR(empty_target_action="error")(
                    preds, targets, indexes=indexes
                ),
                **{
                    f"hit@{k}": RetrievalHitRate(top_k=k, empty_target_action="error")(
                        preds, targets, indexes=indexes
                    )
                    for k in self.hits_at_k
                },
            },
            "preds": preds,
            "targets": targets,
            "indexes": indexes,
        }

        logger.info(
            "for copying...\nversion\tdataset\tmrr\t{}\n{}",
            "\t".join([f"hit@{k}" for k in self.hits_at_k]),
            ",".join(
                [
                    str(version),
                    assert_isinstance(dataloader.dataset, ColumnarDataset).name,
                    "%.4f" % out["result"]["mrr"],
                ]
                + [
                    "%.2f" % (round(float(x) * 100, 2))
                    for x in [out["result"][f"hit@{k}"] for k in self.hits_at_k]
                ]
            ),
        )

        return out

    def transform_eval_batch(self, refs: dict, batch: dict) -> dict:
        n_classes = refs["label_id"].shape[0]
        (batch_size,) = batch["indexes"].shape

        assert batch["column_targets"].shape == (batch_size, n_classes)
        batch["indexes"] = (
            batch["indexes"].view(-1, 1).repeat_interleave(n_classes, dim=1)
        )
        return {k: v.to(self.device) for k, v in batch.items()}


# deprecated
class BaseContrastiveModel(pl.LightningModule, ABC):
    VERSION = 100
    EXPECTED_EVAL_ARGS = set()
    EXPECTED_ARGS = EXPECTED_EVAL_ARGS.union(set())

    @dataclass
    class ContrastiveModelOutput(ModelOutput):
        # scores of negative examples -- higher is better
        neg_scores: torch.Tensor

    def __init__(self, val_dataloader_names: list[str]):
        super().__init__()
        self.hits = []
        self.hits_at_k = [1, 5, 10, 20]

        if len(val_dataloader_names) > 0:
            self.valsets = [f"val_{name}" for name in val_dataloader_names]
        else:
            self.valsets = ["val"]

        # make metrics first
        subsets = ["train", "test"] + self.valsets
        for subset in subsets:
            self.get_metric(subset, "mrr")
            for k in self.hits_at_k:
                self.get_metric(subset, f"hit{k}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self._pl_step(batch, batch_idx, "train")
        return output.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._pl_step(batch, batch_idx, self.valsets[dataloader_idx])

    def test_step(self, batch, batch_idx):
        self._pl_step(batch, batch_idx, "test")

    def _pl_step(self, batch, batch_idx, prefix):
        output = self.forward(**{arg: batch[arg] for arg in self.EXPECTED_ARGS})
        batch_size = output.scores.shape[0]

        # combine probs and negprobs to get the final ranking
        assert output.neg_scores is not None

        indexes: torch.Tensor = batch["indexes"]

        # generate negative indexes
        neg_size = output.neg_scores.shape[1]
        negindexes = indexes.repeat_interleave(neg_size)

        indexes = torch.cat([indexes, negindexes], dim=0)
        preds = torch.cat([output.scores, output.neg_scores.view(-1)], dim=0)
        targets = torch.zeros_like(preds)
        targets[:batch_size] = 1

        self.log(
            f"{prefix}_loss",
            output.loss,
            prog_bar=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

        # compute the loss from
        metric_name = f"{prefix}_mrr"
        metric = self.get_metric(prefix, "mrr")
        metric(preds, targets, indexes=indexes)
        self.log(
            metric_name,
            metric,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
            metric_attribute=metric_name,
            add_dataloader_idx=False,
        )
        for k in self.hits_at_k:
            metric_name = f"{prefix}_hit{k}"
            metric = self.get_metric(prefix, f"hit{k}")
            metric(preds, targets, indexes=indexes)
            self.log(
                metric_name,
                metric,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
                metric_attribute=metric_name,
                add_dataloader_idx=False,
            )
        return output

    def predict(self, dataloader: DataLoader):
        self.eval()

        assert isinstance(dataloader.dataset, ColumnarDataset)
        references = dataloader.dataset.references

        indexes = []
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch = self.transform_eval_batch(references, batch)

                # after transform eval input, we must have "targets" containing 0 - 1 labels
                # of the correct classes and "indexes" containing the id of each example
                assert "targets" in batch and "indexes" in batch

                output = self.forward(
                    **{arg: batch[arg] for arg in self.EXPECTED_EVAL_ARGS}
                )

                preds.append(output.scores)
                targets.append(batch["targets"])
                indexes.append(batch["indexes"])

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        indexes = torch.cat(indexes, dim=0)
        return {"preds": preds, "targets": targets, "indexes": indexes}

    def evaluate(
        self,
        dataloader: DataLoader,
        version: int | str = "-",
    ):
        """Evaluate the performance of the model on the given dataset."""
        predres = self.predict(dataloader)

        preds = predres["preds"]
        targets = predres["targets"]
        indexes = predres["indexes"]

        out = {
            "result": {
                "mrr": RetrievalMRR(empty_target_action="error")(
                    preds, targets, indexes=indexes
                ),
                **{
                    f"hit@{k}": RetrievalHitRate(top_k=k, empty_target_action="error")(
                        preds, targets, indexes=indexes
                    )
                    for k in self.hits_at_k
                },
            },
            "preds": preds,
            "targets": targets,
            "indexes": indexes,
        }

        logger.info(
            "for copying...\nversion\tdataset\tmrr\t{}\n{}",
            "\t".join([f"hit@{k}" for k in self.hits_at_k]),
            ",".join(
                [
                    str(version),
                    assert_isinstance(dataloader.dataset, ColumnarDataset).name,
                    "%.4f" % out["result"]["mrr"],
                ]
                + [
                    "%.2f" % (round(float(x) * 100, 2))
                    for x in [out["result"][f"hit@{k}"] for k in self.hits_at_k]
                ]
            ),
        )

        return out

    def transform_eval_batch(self, refs: dict, batch: dict) -> dict:
        raise NotImplementedError()

    def get_metric(self, subset: str, metric: str):
        attr = f"{subset}_{metric}"
        if not hasattr(self, attr):
            if metric == "mrr":
                setattr(self, attr, RetrievalMRR(empty_target_action="error"))
            else:
                assert metric.startswith("hit")
                topk = int(metric[3:])
                setattr(
                    self,
                    attr,
                    RetrievalHitRate(top_k=topk, empty_target_action="error"),
                )
        return getattr(self, attr)

    @staticmethod
    def format_raw_prediction(
        out: dict,
        dl: DataLoader,
        selected_fields: Optional[list[str]] = None,
        target_field: str = "target_labels",
        use_target_label: bool = True,
    ):
        selected_fields = selected_fields or []
        dataset = assert_isinstance(dl.dataset, ColumnarDataset)

        preds, targets, indexes = (
            out["preds"].cpu().numpy(),
            out["targets"].cpu().numpy(),
            out["indexes"].cpu().numpy(),
        )

        target_df = dataset.references[target_field]
        assert isinstance(target_df, pd.DataFrame)
        if use_target_label:
            target_ids = target_df["label"].tolist()
        else:
            target_ids = target_df.index.tolist()

        indexes = indexes.reshape(-1, len(target_ids))
        preds = preds.reshape(*indexes.shape)
        targets = targets.reshape(*indexes.shape)

        records: list[dict] = [{"index": i} for i in range(indexes.shape[0])]
        if len(records) > 0:
            for field in selected_fields:
                if field in records[0]:
                    raise KeyError(
                        f"field {field} already in record: {list(records[0].keys())}"
                    )

                values = dataset.columns[field]
                if isinstance(values, np.ndarray):
                    for i in range(indexes.shape[0]):
                        records[i][field] = values[i]
                elif isinstance(values, EmbeddingFeat):
                    assert values.decoder is not None
                    for i in range(indexes.shape[0]):
                        records[i][field] = values.decoder[values.value[i]]
                else:
                    raise NotImplementedError(
                        f"Does not support type: {type(values)} yet"
                    )

            for i in range(indexes.shape[0]):
                record = records[i]
                ytrues = np.nonzero(targets[i])[0].tolist()
                if len(ytrues) > 1:
                    record["target"] = [target_ids[x] for x in ytrues]
                else:
                    record["target"] = target_ids[ytrues[0]]

                ypreds = sorted(
                    enumerate(preds[i].tolist()), key=lambda x: x[1], reverse=True
                )
                for j in range(min(5, len(ypreds))):
                    record[f"pred_{j}"] = target_ids[ypreds[j][0]]
                record["correct"] = ypreds[0][0] in ytrues

        return pd.DataFrame(records)
