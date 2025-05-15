from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter

import lightning.pytorch as pl
import torch
import torch.nn as nn
from gpp.sem_label.feat_store import STypeStore
from gpp.sem_label.feat_store_fn import STypeStoreFnArgs
from gpp.sem_label.feats import STFn
from gpp.sem_label.models.base import ModelOutput
from sm.misc.funcs import is_non_decreasing_sequence
from smml.dataset import ColumnarDataset
from torch.utils.data import DataLoader
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from tqdm.auto import tqdm


class BaseContrastiveModelV1(pl.LightningModule, ABC):
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

    def predict(self, dataloader: DataLoader, **kwargs):
        kwargs = self.process_eval_input(**kwargs)

        indexes = []
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch = self.get_eval_input(batch, **kwargs)

                # after get eval input, we must have "targets" containing 0 - 1 labels
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
        **kwargs,
    ):
        """Evaluate the performance of the model on the given dataset."""
        predres = self.predict(dataloader, **kwargs)

        preds = predres["preds"]
        targets = predres["targets"]
        indexes = predres["indexes"]

        return {
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

    def process_eval_input(self, **kwargs) -> dict:
        raise NotImplementedError()

    def get_eval_input(self, batch, **kwargs) -> dict:
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


class STypeEvalModelMixinV1(BaseContrastiveModelV1):
    def make_dataset(
        self,
        store: STypeStore,
        dataset: str,
        text_embedding_model: str,
        n_examples_per_column: int,
    ):
        args = STypeStoreFnArgs(
            dsquery=dataset,
            text_embedding_model=text_embedding_model,
            n_examples_per_column=n_examples_per_column,
            # the rest do not affect when testing data
            n_neg_labels=200,
            neg_sample_spaces="from_dataset",
            min_freq_over_row=0.5,
            max_unmatch_over_ent_row=0.1,
        )

        ds = store(
            args,
            [
                STFn.GetSampleLabelFn,
                STFn.GetHeaderEmbeddingFn,
                STFn.GetNegativeLabelFn,
                STFn.GetTargetLabelExampleEmbeddingFn,
                STFn.GetColumnExampleEmbeddingFn,
            ],
        )

        target_label_examples = ds.columns["GetTargetLabelExampleFn_pos_out"]

        return {
            "ds": self._norm_dataset(ds),
            "example_labels": store.get_func(STFn.GetSampleLabelFn)(args),
            "target_labels": store.get_func(STFn.GetTargetLabelEmbeddingFn)(args),
            "target_label_examples": target_label_examples,
        }

    def predict_dataset(
        self, store, dataset: dict, batch_size: int = 8, verbose: bool = False
    ) -> dict[str, dict[int, dict[str, float]]]:
        dl = DataLoader(dataset["ds"], batch_size=batch_size, num_workers=0)
        predres = self.predict(
            dl,
            example_labels=dataset["example_labels"],
            target_labels=dataset["target_labels"],
            target_label_examples=dataset["target_label_examples"],
        )

        preds = predres["preds"]
        indexes = predres["indexes"].tolist()

        assert is_non_decreasing_sequence(indexes)

        idx2scores = defaultdict(list)
        labels = [e.id for e in dataset["target_labels"].labels]

        for i, x in enumerate(indexes):
            idx2scores[x].append(preds[i].item())

        idx_scores = list(idx2scores.items())
        assert [x[0] for x in idx_scores] == list(range(len(idx_scores)))
        assert len(idx_scores) == len(dataset["example_labels"]["table_id"].value)

        table2output = defaultdict(dict)
        for idx, scores in idx_scores:
            assert len(labels) == len(scores)
            table_id = dataset["example_labels"]["table_id"].value[idx]
            column_index = dataset["example_labels"]["column_index"].value[idx]
            label_score = sorted(zip(labels, scores), key=itemgetter(1), reverse=True)

            table2output[table_id][column_index] = label_score
        return dict(table2output)

    @staticmethod
    def _norm_dataset(ds: ColumnarDataset):
        return ColumnarDataset(
            {
                "indexes": ds.columns["GetSampleLabelFn_example_id"],
                "column_type": ds.columns["GetSampleLabelFn_column_type"],
                "ex_header_embed": ds.columns["GetHeaderEmbeddingFn_column_name_embed"],
                "ex_values_embeds": ds.columns["GetColumnExampleFn_column_values"],
                "ex_values_embeds_mask": ds.columns[
                    "GetColumnExampleFn_column_values_mask"
                ],
                "pos_embeds": ds.columns["GetNegativeLabelFn_pos_labels"],
                "pos_embeds_mask": ds.columns["GetNegativeLabelFn_pos_labels_mask"],
                "neg_embeds": ds.columns["GetNegativeLabelFn_neg_labels"],
                "neg_embeds_mask": ds.columns["GetNegativeLabelFn_neg_labels_mask"],
                "pos_values_embeds": ds.columns["GetTargetLabelExampleFn_pos_labels"],
                "pos_values_embeds_mask": ds.columns[
                    "GetTargetLabelExampleFn_pos_labels_mask"
                ],
                "neg_values_embeds": ds.columns["GetTargetLabelExampleFn_neg_labels"],
                "neg_values_embeds_mask": ds.columns[
                    "GetTargetLabelExampleFn_neg_labels_mask"
                ],
            },
            ds.dtypes,
            ds.collate_fn,
        )


class ContrastiveModelV1(STypeEvalModelMixinV1):
    """Adaptive from resm.distantsupervision.cta_exp.contrastive_v2.

    The difference is that the input parameters are slightly different and
    we predict all semantic labeling -- not just class.
    """

    EXPECTED_EVAL_ARGS = {
        "ex_header_embed",
        "ex_values_embeds",
        "ex_values_embeds_mask",
        "pos_embeds",
        "pos_embeds_mask",
        "pos_values_embeds",
        "pos_values_embeds_mask",
    }
    EXPECTED_ARGS = EXPECTED_EVAL_ARGS.union(
        {
            "neg_embeds",
            "neg_embeds_mask",
            "neg_values_embeds",
            "neg_values_embeds_mask",
        }
    )

    def __init__(
        self, input_dim: int, hidden_dim: int, margin: float, n_hidden_layers: int = 1
    ):
        super().__init__([])

        assert n_hidden_layers >= 1
        self.margin = margin
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.header_transformation = nn.Sequential(
            *(
                [nn.Linear(input_dim, hidden_dim)]
                + [
                    x
                    for lst in [
                        [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
                        for _ in range(n_hidden_layers - 1)
                    ]
                    for x in lst
                ]
            )
        )

        self.value_transformation = nn.Sequential(
            *(
                [nn.Linear(input_dim, hidden_dim)]
                + [
                    x
                    for lst in [
                        [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
                        for _ in range(n_hidden_layers - 1)
                    ]
                    for x in lst
                ]
            )
        )

        self.save_hyperparameters()

    def calculate_distance(self, embed1, embed2) -> torch.Tensor:
        """Calculate the distance between two embeddings. Returns a tensor of shape (batch_size,).

        The distance should be a positive number, where a smaller distance means a higher similarity.
        """
        return torch.norm(embed1 - embed2, dim=-1)

    def distance_to_score(self, distance: torch.Tensor) -> torch.Tensor:
        """Convert a distance to a score, in which higher score is better (for distance, lower is better)"""
        return -distance

    def forward(
        self,
        ex_header_embed,
        ex_values_embeds,
        ex_values_embeds_mask,
        pos_embeds,
        pos_embeds_mask,
        pos_values_embeds,
        pos_values_embeds_mask,
        neg_embeds=None,
        neg_embeds_mask=None,
        neg_values_embeds=None,
        neg_values_embeds_mask=None,
    ) -> BaseContrastiveModelV1.ContrastiveModelOutput:
        ex_header_embed = self.header_transformation(ex_header_embed)
        pos_embeds = self.header_transformation(pos_embeds)

        # ex values embed: (batch_size, n_values, embed_size)
        ex_values_embeds = self.value_transformation(ex_values_embeds)
        # mean-pooling over the examples
        ex_values_embeds = ex_values_embeds * ex_values_embeds_mask.unsqueeze(-1)
        ex_values_embeds = ex_values_embeds.sum(dim=-2) / ex_values_embeds_mask.sum(
            dim=-1, keepdim=True
        )

        # pos values embed: (batch_size, n_classes, n_examples, embed_size)
        pos_values_embeds = self.value_transformation(pos_values_embeds)
        pos_values_embeds = pos_values_embeds * pos_values_embeds_mask.unsqueeze(-1)
        # pos values embed: (batch_size, n_classes, embed_size) -- some classes have zero examples (invalid)
        pos_values_embeds = pos_values_embeds.sum(dim=-2) / pos_values_embeds_mask.sum(
            dim=-1, keepdim=True
        )

        # new_ex_embed = (batch_size, embed_size * 2)
        new_ex_embed = torch.cat([ex_header_embed, ex_values_embeds], dim=-1)
        # pos_embeds: (batch_size, n_classes, embed_size * 2)
        new_pos_embeds = torch.cat([pos_embeds, pos_values_embeds], dim=-1)
        # outofrange_mask (batch_size, n_classes)
        pos_outofrange_mask = (1 - pos_embeds_mask) * 100  #
        # pos_dis: (batch size, )
        pos_dis, _ = torch.min(
            self.calculate_distance(
                new_ex_embed.unsqueeze(dim=1),
                new_pos_embeds,
            )
            * pos_embeds_mask
            + pos_outofrange_mask,
            dim=-1,
        )
        scores = self.distance_to_score(pos_dis)

        if neg_embeds is not None:
            assert (
                neg_embeds_mask is not None
                and neg_values_embeds is not None
                and neg_values_embeds_mask is not None
            )

            # sample the negative examples
            # neg_embeds = neg_embeds[:, :, :]
            # mask_neg_classes = mask_neg_classes[:, :]
            # assert torch.all(mask_neg_classes == 1)

            # neg_values_embeds has size: batch, n_classes, n_examples, embed_size
            neg_values_embeds = self.value_transformation(neg_values_embeds)
            neg_values_embeds = neg_values_embeds * neg_values_embeds_mask.unsqueeze(-1)

            # mean-pooling over the examples (batch, n_classes, embed_size)
            neg_values_embeds = neg_values_embeds.sum(
                dim=-2
            ) / neg_values_embeds_mask.sum(dim=-1, keepdim=True)

            neg_embeds = self.header_transformation(neg_embeds)
            # new_neg_embeds = (batch size, n neg classes, embed_size * 2)
            new_neg_embeds = torch.cat(
                [
                    neg_embeds,
                    neg_values_embeds,
                ],
                dim=-1,
            )
            neg_outofrange_mask = (1 - neg_embeds_mask) * 100  #
            # neg_dis: (batch size, n neg classes)
            neg_dis = self.calculate_distance(
                new_ex_embed.unsqueeze(dim=1),
                new_neg_embeds,
            )
            loss = torch.maximum(
                (pos_dis.view(-1, 1) - neg_dis + self.margin) * neg_embeds_mask,
                torch.zeros_like(pos_dis.view(-1, 1)),
            )
            loss = torch.sum(loss) / torch.sum(neg_embeds_mask)

            # mask out the negative examples that are not valid
            # by assigning a large negative score to ensure they are not selected
            neg_scores = self.distance_to_score(
                neg_dis * neg_embeds_mask + neg_outofrange_mask
            )
        else:
            loss = torch.tensor(0.0)
            neg_scores = torch.tensor(0.0)

        return BaseContrastiveModelV1.ContrastiveModelOutput(
            loss=loss, scores=scores, neg_scores=neg_scores
        )

    def process_eval_input(
        self,
        example_labels: dict,
        target_labels,
        target_label_examples,
    ) -> dict:
        ex_embs = target_label_examples.get_ex_embeddings()
        pos_values_embeds = ex_embs.embeddings[ex_embs.label_examples]
        return {
            "original_column_type": example_labels["original_column_type"].value,
            "target_label_index": target_labels.id2index,
            "pos_embeds": torch.from_numpy(target_labels.embeddings),
            "pos_values_embeds": torch.from_numpy(pos_values_embeds),
            "pos_values_embeds_mask": torch.from_numpy(
                target_label_examples.mask_label_examples
            ),
        }

    def get_eval_input(
        self,
        batch,
        original_column_type,
        target_label_index,
        pos_embeds,
        pos_values_embeds,
        pos_values_embeds_mask,
    ):
        batch_size = batch["indexes"].shape[0]
        n_classes = pos_embeds.shape[0]
        out = {}
        for key in ["ex_header_embed", "ex_values_embeds", "ex_values_embeds_mask"]:
            out[key] = batch[key].repeat_interleave(n_classes, dim=0)

        out["pos_embeds"] = pos_embeds.repeat(batch_size, 1).unsqueeze(1)
        out["pos_embeds_mask"] = torch.ones(out["pos_embeds"].shape[:-1])
        out["pos_values_embeds"] = pos_values_embeds.repeat(batch_size, 1, 1).unsqueeze(
            1
        )
        out["pos_values_embeds_mask"] = pos_values_embeds_mask.repeat(
            batch_size, 1
        ).unsqueeze(1)

        out["indexes"] = batch["indexes"].repeat_interleave(n_classes, dim=0)
        # create one hot vectors for the targets
        targets = torch.zeros(
            (batch_size, n_classes), dtype=torch.long, device=self.device
        )
        for i, xs in enumerate(batch["column_type"]):
            for x in original_column_type[xs]:
                targets[i, target_label_index[x]] = 1
        out["targets"] = targets.view(-1)

        for k, v in out.items():
            out[k] = v.to(self.device)
        return out
