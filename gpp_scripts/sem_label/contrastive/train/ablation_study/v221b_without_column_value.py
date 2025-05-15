from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import lightning as L
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpp.sem_label.models.base import ModelOutput, V210BaseContrastiveLearningModel
from loguru import logger
from sm.misc.funcs import assert_isinstance
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from transformers import AutoModel, BertForSequenceClassification, BertPreTrainedModel


class V221b(V210BaseContrastiveLearningModel):
    VERSION = 101
    EXPECTED_EVAL_ARGS = {
        "page_title",
        "table_header",
        "column_header",
        "column_values",
        "dataset_id",
    }
    EXPECTED_ARGS = EXPECTED_EVAL_ARGS.union({"column_targets", "column_targets_mask"})

    def __init__(
        self,
        embedding_dim: int,
        compress_dim: int,
        margin: float,
        sim_metric: Literal["cosine", "vecnorm"],
        norm_input_embedding: bool,
        val_dataloader_names: Optional[list[str]] = None,
    ):
        super().__init__(
            val_dataloader_names or [],
            ["mrr", "hit"],
        )

        self.compressor = nn.Sequential(
            nn.Linear(embedding_dim, compress_dim)
            # nn.LazyBatchNorm1d(),
            # nn.GELU(),
        )

        self.margin = margin
        self.norm_input_embedding = norm_input_embedding
        self.sim_metric = sim_metric
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        print("Optimizer:", optimizer)
        return optimizer

    def calculate_distance(self, embed1, embed2) -> torch.Tensor:
        """Calculate the distance between two embeddings. Returns a tensor of shape (batch_size,).

        The distance should be a positive number, where a smaller distance means a higher similarity.
        """
        if self.sim_metric == "vecnorm":
            return torch.norm(embed1 - embed2, dim=-1)
        if self.sim_metric == "cosine":
            return 1 - F.cosine_similarity(embed1, embed2, dim=-1)
        raise ValueError(f"Unknown metric: {self.sim_metric}")

    def distance_to_score(self, distance: torch.Tensor) -> torch.Tensor:
        """Convert a distance to a score, in which higher score is better (for distance, lower is better)"""
        if self.sim_metric == "vecnorm":
            return -distance
        if self.sim_metric == "cosine":
            return 1 - distance
        raise ValueError(f"Unknown metric: {self.sim_metric}")

    def forward(
        self,
        page_title: torch.Tensor,
        table_header: torch.Tensor,
        column_header: torch.Tensor,
        column_values: torch.Tensor,
        dataset_id: torch.Tensor,
        column_targets: Optional[torch.Tensor] = None,
        column_targets_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        # label_name: C x D2
        label_name, label_desc, label_exs = self.get_label_input(
            assert_isinstance(dataset_id[0].item(), int)
        )

        if self.norm_input_embedding:
            # page_title = F.normalize(page_title, dim=-1)
            # table_header = F.normalize(table_header, dim=-1)
            column_header = F.normalize(column_header, dim=-1)
            column_values = F.normalize(column_values, dim=-1)
            label_name = F.normalize(label_name, dim=-1)
            # label_desc = F.normalize(label_desc, dim=-1)
            label_exs = F.normalize(label_exs, dim=-1)

            # page_title = self.compressor(page_title)
            # table_header = self.compressor(F.normalize(table_header, dim=-1))

        # B x D1
        column_header = self.compressor(column_header)
        column_values = self.compressor(column_values)

        # C x D2
        label_name = self.compressor(label_name)
        # label_desc = self.compressor(label_desc)
        label_exs = self.compressor(label_exs)

        b, d1 = column_header.shape
        c, d2 = label_name.shape
        assert d1 == d2

        # make it B x C x D1
        expanded_column_header = (
            column_header.view(b, 1, d1).expand(b, c, d1).reshape(b * c, d1)
        )
        expanded_column_values = (
            column_values.view(b, 1, d1).expand(b, c, d1).reshape(b * c, d1)
        )
        expanded_label_name = (
            label_name.view(1, c, d2).expand(b, c, d2).reshape(b * c, d2)
        )
        expanded_label_exs = (
            label_exs.view(1, c, d2).expand(b, c, d2).reshape(b * c, d2)
        )

        # (B x C) -- 1 dim
        header_dis = self.calculate_distance(
            expanded_column_header, expanded_label_name
        )
        value_dis = self.calculate_distance(expanded_column_values, expanded_label_exs)

        assert header_dis.shape == (b * c,)
        assert value_dis.shape == (b * c,)

        combine_dis = header_dis
        # combine_dis = self.calculate_distance(col_emb, label_emb)
        # assert combine_dis.shape == (b * c,)

        if column_targets is not None:
            # column targets is B x C
            # make negative and positive embeddings from this
            assert column_targets_mask is not None
            pos_idx, neg_idx = self.make_contrastive_examples(
                column_targets, column_targets_mask
            )
            header_loss = torch.mean(
                torch.maximum(
                    header_dis[pos_idx] - header_dis[neg_idx] + self.margin,
                    torch.zeros_like(pos_idx),
                )
            )

            value_loss = torch.mean(
                torch.maximum(
                    value_dis[pos_idx] - value_dis[neg_idx] + self.margin,
                    torch.zeros_like(pos_idx),
                )
            )

            combine_loss = torch.mean(
                torch.maximum(
                    combine_dis[pos_idx] - combine_dis[neg_idx] + self.margin,
                    torch.zeros_like(pos_idx),
                )
            )

            # loss = header_loss + value_loss + combine_loss * 2
            loss = combine_loss
        else:
            loss = torch.tensor(0.0)

        return ModelOutput(
            loss=loss, scores=self.distance_to_score(combine_dis).view(b, c)
        )

    def make_contrastive_examples(
        self, column_targets: torch.Tensor, column_targets_mask: torch.Tensor
    ):
        batchsize, nclasses = column_targets.shape
        idx = torch.arange(batchsize * nclasses, device=column_targets.device).view(
            batchsize, nclasses
        )

        out_pos_idx = []
        out_neg_idx = []

        for i in range(batchsize):
            sidx = idx[i]

            pos_idx = sidx[column_targets[i] == 1]
            neg_idx = sidx[
                torch.logical_and(column_targets[i] == 0, column_targets_mask[i] == 1)
            ]

            npos, nneg = pos_idx.shape[0], neg_idx.shape[0]
            pos_idx = pos_idx.repeat_interleave(nneg)
            neg_idx = neg_idx.repeat(npos)

            out_pos_idx.append(pos_idx)
            out_neg_idx.append(neg_idx)

        out_pos_idx = torch.cat(out_pos_idx, dim=0)
        out_neg_idx = torch.cat(out_neg_idx, dim=0)
        return out_pos_idx, out_neg_idx
