from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import orjson
import pandas as pd
import serde.jl
import serde.pickle
from gpp.sem_label.data_modules.base import BaseDataModule, PrecomputedEmbeddingMixin
from keyvec import EmbeddingManager
from libactor.cache import BackendFactory, cache
from sm.prelude import I, M
from smml.data_model_helper import (
    EncodedSingleNumpyArray,
    Single2DNumpyArray,
    SingleNDNumpyArray,
    SingleNumpyArray,
)
from smml.dataset import ColumnarDataset, EmbeddingFeat, extended_collate_fn
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class SLabelV210DataModule(BaseDataModule, PrecomputedEmbeddingMixin):
    VERSION = 104

    def __init__(
        self,
        data_dir: Path | str,
        model_name_or_path: str,
        embedding_manager: EmbeddingManager,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        text_delimiter: str = ", ",
        n_examples_per_column: int = 100,
        n_examples_per_label: int = 150,
    ):
        super().__init__(
            data_dir=data_dir,
            params={
                "model_name_or_path": model_name_or_path,
                "delimiter": text_delimiter,
                "n_examples_per_column": n_examples_per_column,
                "n_examples_per_label": n_examples_per_label,
                "embedding_manager": embedding_manager.embedding_model.get_args().to_dict(),
            },
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )

        self.model_name_or_path = model_name_or_path
        self.text_delimiter = text_delimiter
        self.n_examples_per_column = n_examples_per_column
        self.n_examples_per_label = n_examples_per_label
        self.embedding_manager = embedding_manager

        self.disable_cache = False

        self.dataset_ids: dict[str, int] = {
            "wt-limited-easy-sp51-v1": 0,
            "wt250": 1,
            "t2dv2": 2,
        }

    def setup(self, stage: str):
        if stage == "fit":
            self.train = self.load_dataset("wt-limited-easy-sp51-v1")
            self.dev = [
                self.load_dataset("wt250"),
                self.load_dataset("t2dv2"),
            ]
        elif stage == "validate":
            self.dev = [
                self.load_dataset("wt250"),
                self.load_dataset("t2dv2"),
            ]
        elif stage == "test":
            self.test = [
                self.load_dataset("wt250"),
                self.load_dataset("t2dv2"),
            ]
        else:
            raise NotImplementedError(stage)

    @cache(backend=BackendFactory.actor.mem)
    def load_dataset(self, name: str) -> ColumnarDataset:
        dataset = self.load_transformed_dataset(name)
        dataset = self.make_columnar_dataset(name, dataset)
        return dataset

    @cache(BackendFactory.actor.mem)
    def load_raw_dataset(self, name: str):
        return serde.pickle.deser(self.data_dir / f"{name}.pkl")

    # @cache(
    #     backend=wrap_backend(
    #         DirBackend(
    #             ser=ser_dict_array,
    #             deser=deser_dict_array,
    #             dirname=lambda self, name, is_train: f"transformed/{name}"
    #             + ("_train" if is_train else ""),
    #         ),
    #         mem_persist=True,
    #         log_serde_time=True,
    #     ),
    #     disable="disable_cache",
    # )
    def load_transformed_dataset(self, name: str):
        dataset = self.load_raw_dataset(name)
        return self.transformation(name, dataset)

    def transformation(self, name: str, dataset: dict):
        max_text_len = self.get_max_text_len()
        target2index = self.get_target2index(
            dataset["target_labels"], is_cta_only=False
        )

        label_ids = [""] * len(target2index)
        label_names = [""] * len(target2index)
        label_descs = [""] * len(target2index)
        label_examples = [""] * len(target2index)
        for target, index in target2index.items():
            out = self.convert_label(
                target,
                dataset["target_labels"],
                dataset["target_label_examples"],
                delimiter=self.text_delimiter,
                max_n_examples=self.n_examples_per_label,
            )
            label_ids[index] = target
            label_names[index] = out["label"]
            label_descs[index] = out["description"]
            label_examples[index] = out["examples"]

        lst = []
        for sample in tqdm(dataset["samples"], desc="processing"):
            out = self.convert_table(
                sample,
                dataset["target_labels"],
                target2index,
                max_text_len,
                text_delimiter=self.text_delimiter,
                max_n_examples=self.n_examples_per_column,
            )
            lst.append(out)

        n_samples = sum(len(x["column_header"]) for x in lst)

        table_ids = np.empty(n_samples, dtype=np.object_)
        page_titles = np.empty(n_samples, dtype=np.object_)
        table_headers = np.empty(n_samples, dtype=np.object_)
        column_indices = np.empty(n_samples, dtype=np.int32)
        column_headers = np.empty(n_samples, dtype=np.object_)
        column_values = np.empty(n_samples, dtype=np.object_)

        # N x C -- where C is the number of classes
        column_targets = np.zeros((n_samples, len(target2index)), dtype=np.int32)
        column_targets_mask = np.zeros_like(column_targets)

        start = 0
        for i, out in enumerate(lst):
            end = start + len(out["column_header"])

            table_ids[start:end] = out["table_id"]
            page_titles[start:end] = out["page_title"] or ""
            table_headers[start:end] = out["table_header"]
            column_indices[start:end] = out["column_index"]
            column_headers[start:end] = out["column_header"]
            column_values[start:end] = out["column_values"]

            column_targets[start:end] = out["encoded_column_type"]
            column_targets_mask[start:end] = out["encoded_column_type_mask"]

            start = end

        return {
            "sample_id": SingleNumpyArray(np.arange(table_ids.shape[0])),
            "table_id": SingleNumpyArray(table_ids),
            "page_title": EncodedSingleNumpyArray.from_array(page_titles),
            "table_header": EncodedSingleNumpyArray.from_array(table_headers),
            "column_index": SingleNumpyArray(column_indices),
            "column_header": EncodedSingleNumpyArray.from_array(column_headers),
            "column_values": EncodedSingleNumpyArray.from_array(column_values),
            "label_id": SingleNumpyArray(np.array(label_ids, dtype=np.object_)),
            "label_name": EncodedSingleNumpyArray.from_array(
                np.array(label_names, dtype=np.object_)
            ),
            "label_desc": EncodedSingleNumpyArray.from_array(
                np.array(label_descs, dtype=np.object_)
            ),
            "label_example": EncodedSingleNumpyArray.from_array(
                np.array(label_examples, dtype=np.object_)
            ),
            "column_targets": SingleNDNumpyArray(column_targets),
            "column_targets_mask": SingleNDNumpyArray(column_targets_mask),
        }

    def make_columnar_dataset(
        self,
        name: str,
        dataset: dict,
        embedding_readonly: bool = True,
    ):
        if embedding_readonly:
            embed_fn = self.embedding_manager.batch_retrieve_exist
        else:
            embed_fn = partial(self.embedding_manager.batch_get, batch_size=8)

        if name not in self.dataset_ids:
            self.dataset_ids[name] = len(self.dataset_ids)

        columnar_dataset = ColumnarDataset(
            columns={
                "indexes": dataset["sample_id"].value,
                "page_title": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["page_title"], embed_fn
                ),
                "table_header": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["table_header"], embed_fn
                ),
                "column_header": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["column_header"], embed_fn
                ),
                "column_values": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["column_values"], embed_fn
                ),
                "column_targets": dataset["column_targets"].value,
                "column_targets_mask": dataset["column_targets_mask"].value,
                "dataset_id": np.full(
                    dataset["sample_id"].value.shape,
                    self.dataset_ids[name],
                    dtype=np.int32,
                ),
            },
            references={
                "dataset": dataset,
                "table_id": dataset["table_id"].value,
                "col_index": dataset["column_index"].value,
                "label_id": dataset["label_id"].value,
                "label_name": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["label_name"], embed_fn
                ),
                "label_desc": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["label_desc"], embed_fn
                ),
                "label_example": EmbeddingFeat.from_encoded_single_numpy_array(
                    dataset["label_example"], embed_fn
                ),
            },
            dtypes={"column_targets_mask": np.bool_},
            collate_fn=extended_collate_fn,
            name=name,
        )

        if not embedding_readonly:
            self.embedding_manager.flush(soft=True)
        return columnar_dataset

    @cache(BackendFactory.actor.mem)
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer

    def get_max_text_len(self):
        tokenizer = self.get_tokenizer()
        model_max_len = tokenizer.model_max_length
        if model_max_len == 512:
            return 800
        elif model_max_len == 8192:
            return 10240
        else:
            raise NotImplementedError(model_max_len)

    def convert_table(
        self,
        sample: dict,
        target_labels: pd.DataFrame,
        target2index: dict[str, int],
        max_text_len: int,
        text_delimiter: str = ", ",
        max_n_examples: int = 150,
    ):
        tbl = sample["table"]
        context = sample["context"]
        # no remove columns
        assert list(range(len(tbl.columns))) == [col.index for col in tbl.columns]

        column_index: list[int] = sample["column_index"]
        column_types: list[list[str]] = sample["column_type"]

        ignore_cols: set[int] = {
            col.index
            for col in tbl.columns
            if M.assert_not_null(col.clean_name).strip().isdigit()
        }
        tmp = len(ignore_cols)
        ignore_cols = ignore_cols.difference(column_index)
        # if len(ignore_cols) != tmp:
        #     raise Exception()
        # assert all(i not in ignore_cols for i in column_index)

        df = tbl.df
        column_texts = []
        for ci in column_index:
            coldata = df[df.columns[ci]].drop_duplicates()[:max_n_examples]
            column_text = self.ensure_text_len(
                lambda x: self.list_to_text(coldata[:x], delimiter=text_delimiter),
                max_text_len,
            )
            column_texts.append(column_text)

        # multi-class version - K x C where K is the number of columns and C is the number of classes
        enc_targ = np.zeros((len(column_index), len(target2index)), dtype=np.int32)
        enc_targ_mask = np.zeros_like(enc_targ)

        # binary version - each sample is a pair of column and target as well as a label 0 or 1
        bin_cols = []
        bin_targ = []
        bin_lbls = []
        for cidx, ci, pos_ids in zip(
            range(len(column_index)), column_index, column_types
        ):
            sim_pos_ids = set(pos_ids)
            for id in pos_ids:
                sim_pos_ids = sim_pos_ids.union(
                    (
                        k
                        for k, v in target_labels.loc[id].similar_labels.items()
                        if v <= 1
                    )
                )

            neg_ids = [x for x in target2index.keys() if x not in sim_pos_ids]

            pos_ids_index = [target2index[x] for x in pos_ids]
            neg_ids_index = [target2index[x] for x in neg_ids]

            enc_targ[cidx, pos_ids_index] = 1
            enc_targ_mask[cidx, pos_ids_index] = 1
            enc_targ_mask[cidx, neg_ids_index] = 1

            for pos_id_idx in pos_ids_index:
                bin_cols.append(ci)
                bin_targ.append(pos_id_idx)
                bin_lbls.append(1)
            for neg_id_idx in neg_ids_index:
                bin_cols.append(ci)
                bin_targ.append(neg_id_idx)
                bin_lbls.append(0)

        return {
            "table_id": tbl.table_id,
            "page_title": context.page_title,
            "table_header": text_delimiter.join(
                [col.clean_multiline_name or "" for col in tbl.columns]
            ),
            "column_index": column_index,
            "column_header": [
                tbl.get_column_by_index(ci).clean_multiline_name or ""
                for ci in column_index
            ],
            "column_values": column_texts,
            "encoded_column_type": enc_targ,
            "encoded_column_type_mask": enc_targ_mask,
        }

    def list_to_text(self, lst: list, delimiter: str = ", "):
        norm_lst = []
        for item in lst:
            if isinstance(item, str):
                item = item.replace("\n", " ").strip()
                if len(item) > 0:
                    norm_lst.append(item)
            elif isinstance(item, (int, bool)):
                norm_lst.append(str(item))
            elif isinstance(item, float):
                if math.isnan(item):
                    continue
                x = str(item)
                a, b = x.split(".")
                norm_lst.append(f"{a}.{b[:4]}")
            elif item is None:
                continue
            else:
                assert False, (item, type(item))
        return delimiter.join(norm_lst)

    def ensure_text_len(self, fn: Callable[[Optional[int]], str], max_len: int) -> str:
        text = fn(None)
        i = -1
        while len(text) > max_len:
            text = fn(i)
            i -= 1
        return text

    def convert_label(
        self,
        label: str,
        target_labels: pd.DataFrame,
        target_label_examples: dict[str, list[str]],
        delimiter: str,
        max_n_examples: Optional[int] = 150,
    ):
        obj = target_labels.loc[label]
        max_text_len = self.get_max_text_len()
        return {
            "label": obj.label,
            "description": obj.description,
            "examples": self.ensure_text_len(
                lambda i: self.list_to_text(
                    target_label_examples[label][:max_n_examples][:i], delimiter
                ),
                max_text_len,
            ),
        }

    def get_target2index(self, target_labels: pd.DataFrame, is_cta_only: bool):
        if is_cta_only:
            target_label_ids = [
                x for x in target_labels.index if not target_labels.loc[x].is_prop
            ]
        else:
            target_label_ids = target_labels.index

        return dict(zip(target_label_ids, range(len(target_label_ids))))

    def make_sample_id(self, start: int, indexes: np.ndarray):
        """Convert an array of consecutive column index of a table to sample id"""
        uniq_val, uniq_idx, uniq_count = np.unique(
            indexes, return_index=True, return_counts=True
        )

        # by repeat the group id by number of samples per group, we will get back the indexes array -- only if samples
        # from same group are next to each other.
        order = np.argsort(
            uniq_idx
        )  # reorder the first index that the group id appears
        assert np.all(
            uniq_val[order].repeat(uniq_count[order]) == indexes
        ), "samples from the same group are next to each other"

        return (np.arange(uniq_val.shape[0]) + start)[order].repeat(
            uniq_count[order]
        ), uniq_val.shape[0] + start

    def filter_no_groundtruth(self, ds: dict):
        """Remove samples that do not have correct entities"""
        indexes = ds["sample_id"].value
        targets = ds["binary_column_type_label"].value

        # *** STEP 1 ***
        # we want to check that samples from the same group are next to each other
        uniq_val, uniq_idx, uniq_count = np.unique(
            indexes, return_index=True, return_counts=True
        )

        # by repeat the group id by number of samples per group, we will get back the indexes array -- only if samples
        # from same group are next to each other.
        order = np.argsort(
            uniq_idx
        )  # reorder the first index that the group id appears -- this is due to np.unique returns a sorted array
        assert np.all(
            uniq_val[order].repeat(uniq_count[order]) == indexes
        ), "samples from the same group are next to each other"

        # create a mask to filter out samples
        sample_ranges = np.append(uniq_idx[order], indexes.shape[0])
        mask = np.ones_like(targets, dtype=np.bool_)
        for i in range(sample_ranges.shape[0] - 1):
            gi, gj = sample_ranges[i], sample_ranges[i + 1]
            n_label = targets[gi:gj].sum()
            # assert n_label <= 1, (gi, gj, n_label)
            if n_label == 0:
                mask[gi:gj] = 0

        out = {}
        for k, v in ds.items():
            if isinstance(v, SingleNumpyArray):
                out[k] = SingleNumpyArray(v.value[mask])
            elif isinstance(v, EncodedSingleNumpyArray):
                out[k] = EncodedSingleNumpyArray(v.decoder, v.value[mask])
            elif isinstance(v, Single2DNumpyArray):
                out[k] = Single2DNumpyArray(v.value[mask])
            else:
                raise NotImplementedError(type(v))

        return out


def transformed_dataset_to_pd(
    dataset: dict, keys: Optional[list[str]] = None, size: int = 200
):
    if keys is None:
        keys = list(dataset.keys())

    out = {}
    for key in keys:
        if isinstance(dataset[key], SingleNumpyArray):
            out[key] = dataset[key].value[:size]
        elif isinstance(dataset[key], Single2DNumpyArray):
            out[key] = dataset[key].value[:size]
        elif isinstance(dataset[key], SingleNDNumpyArray):
            out[key] = dataset[key].value[:size]
        elif isinstance(dataset[key], EncodedSingleNumpyArray):
            out[key] = [dataset[key].decoder[x] for x in dataset[key].value[:size]]
        else:
            raise NotImplementedError(type(dataset[key]))

        out[key] = [str(x) for x in out[key]]

    return pd.DataFrame(out)
