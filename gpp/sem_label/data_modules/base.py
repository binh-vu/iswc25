from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import lightning as L
import lightning.pytorch as pl
import numpy as np
import orjson
import pandas as pd
import serde.jl
import torch
from IPython.display import display
from keyvec import EmbeddingManager, HfModelArgs
from loguru import logger
from pytorch_lightning.loggers import CSVLogger
from sm.misc.fn_cache import CacheMethod
from sm.misc.funcs import assert_isinstance
from sm.prelude import I, M
from smml.data_model_helper import (
    EncodedSingleMasked2DNumpyArray,
    EncodedSingleNumpyArray,
    Single2DNumpyArray,
    SingleNDNumpyArray,
    SingleNumpyArray,
    SinglePandasDataFrame,
    deser_dict_array,
    ser_dict_array,
)
from smml.dataset import ColumnarDataset
from timer import Timer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def get_best_model(
    csv_log_dir: Path | str,
    metric: str,
    smaller_is_better: bool = True,
    restore_step: Optional[int] = None,
):
    csv_log_dir = Path(csv_log_dir)
    df = pd.read_csv(csv_log_dir / "metrics.csv")
    if restore_step is not None:
        ser = df.loc[df[df["step"] == restore_step - 1].index[0]]
    else:
        if smaller_is_better:
            ser = df.loc[df[df[metric].notna()][metric].idxmin()]
        else:
            ser = df.loc[df[df[metric].notna()][metric].idxmax()]
    ckpt_file = (
        csv_log_dir
        / f"checkpoints/epoch={int(ser['epoch'])}-step={int(ser['step']) + 1}.ckpt"
    )
    assert ckpt_file.exists(), ckpt_file
    return ckpt_file


def load_best_model(trainer: pl.Trainer, metric: str, smaller_is_better: bool = True):
    return trainer.model.__class__.load_from_checkpoint(
        get_best_model(
            assert_isinstance(trainer.logger, CSVLogger).log_dir,
            metric,
            smaller_is_better,
        )
    )


class BaseDataModule(L.LightningDataModule):
    VERSION = 100

    def __init__(
        self,
        data_dir: Path | str,
        params: dict,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.params = params
        self.logger = logger.bind(name=self.__class__.__name__)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train: Optional[ColumnarDataset] = None
        self.dev: Optional[ColumnarDataset | Sequence[ColumnarDataset]] = None
        self.test: Optional[ColumnarDataset | Sequence[ColumnarDataset]] = None

    # def get_working_fs(self) -> FS:
    #     if not hasattr(self, "working_fs"):
    #         cache_dir = ReamWorkspace(
    #             self.data_dir / "data_modules"
    #         ).reserve_working_dir(
    #             ActorState.create(self.__class__, self.params, dependencies=[])
    #         )
    #         self.logger.debug("Using working directory: {}", cache_dir)
    #         self.working_fs = FS(cache_dir)
    #     return self.working_fs

    def train_dataloader(self):
        assert self.train is not None
        return DataLoader(
            self.train, batch_size=self.train_batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        assert self.dev is not None
        if isinstance(self.dev, list):
            return [
                DataLoader(dev, batch_size=self.eval_batch_size, pin_memory=True)
                for dev in self.dev
            ]
        else:
            assert self.dev is not None
            return DataLoader(
                self.dev, batch_size=self.eval_batch_size, pin_memory=True
            )

    def get_dataloader(self, dataset: ColumnarDataset):
        return DataLoader(dataset, batch_size=self.eval_batch_size, pin_memory=True)


class PrecomputedEmbeddingMixin:
    embedding_manager: EmbeddingManager
    dataset_names: list[str]
    dataset_ids: dict[str, int]

    def load_transformed_dataset(self, dataset: str, is_train: bool) -> dict:
        raise NotImplementedError()

    def make_embeddings(self, batch_size: int = 64) -> None:
        # fetch text embeddings
        text_emb = self.embedding_manager
        texts = set()

        for dataset in self.dataset_names:
            logger.info("[prepare data] loading transformed dataset:  {}", dataset)
            ds = self.load_transformed_dataset(dataset, is_train=True)
            for key, val in ds.items():
                if isinstance(val, EncodedSingleNumpyArray):
                    texts = texts.union(val.decoder)

        texts = [
            text
            for text in tqdm(texts, desc="filter precomputed embeddings")
            if text not in text_emb
        ]
        if len(texts) > 0:
            text_emb.batch_get(texts, batch_size=batch_size, verbose=True)
            text_emb.flush()

    def get_embedding_dim(self):
        embedding_model = assert_isinstance(
            self.embedding_manager.embedding_model.get_args(), HfModelArgs
        ).embedding_model
        if embedding_model == "BAAI/bge-m3":
            return 1024
        if embedding_model == "sentence-transformers/all-mpnet-base-v2":
            return 768
        raise NotImplementedError(
            f"Not implement for the embedding model: {embedding_model}"
        )
