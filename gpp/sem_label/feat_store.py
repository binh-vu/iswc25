from functools import lru_cache
from typing import Mapping, TypeVar

import numpy as np
from gp.misc.itemdistance import KGItemDistance
from gpp.sem_label.feat_store_fn import STypeStoreFn, STypeStoreFnArgs
from keyvec import EmbeddingManager, HfModelArgs
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from libactor.actor import Actor
from libactor.cache import BackendFactory, cache
from libactor.misc import NoParams
from libactor.storage import GlobalStorage
from loguru import logger
from slugify import slugify
from smml.data_model_helper import (
    DictNumpyArray,
    Single2DNumpyArray,
    SingleLevelIndexedPLDataFrame,
    SingleNumpyArray,
)
from smml.dataset import ColumnarDataset, Feat, extended_collate_fn

Fn = TypeVar("Fn", bound=STypeStoreFn)


class STypeStore(Actor[NoParams]):
    VERSION = 103

    def __call__(self, args: STypeStoreFnArgs, enable_fns: list[type[STypeStoreFn]]):
        with STypeStoreFn.auto_clear_mem_backend():
            columns: dict[str, list | np.ndarray | Feat] = {}

            for fn in enable_fns:
                value = self.get_func(fn)(args)
                if isinstance(value, (SingleNumpyArray, Single2DNumpyArray)):
                    value = value.value
                    assert isinstance(value, np.ndarray), type(value)
                    columns[self.get_func_name(fn)] = value
                elif isinstance(value, SingleLevelIndexedPLDataFrame):
                    for col in value.value.columns:
                        columns[self.get_func_name(fn) + f"_{col}"] = value.value[
                            col
                        ].to_numpy()
                elif isinstance(value, DictNumpyArray):
                    for col, value in value.value.items():
                        columns[self.get_func_name(fn) + f"_{col}"] = value
                elif isinstance(value, Feat):
                    columns[self.get_func_name(fn)] = value
                elif isinstance(value, dict):
                    for col, value in value.items():
                        if isinstance(value, (SingleNumpyArray, Single2DNumpyArray)):
                            value = value.value
                            assert isinstance(value, np.ndarray), type(value)
                        else:
                            assert isinstance(value, (Feat, np.ndarray)), type(value)
                        columns[self.get_func_name(fn) + f"_{col}"] = value
                else:
                    raise TypeError(type(value))

            return ColumnarDataset(columns, collate_fn=extended_collate_fn)

    def invoke_fns(self, args: STypeStoreFnArgs, fns: list[type[STypeStoreFn]]):
        with STypeStoreFn.auto_clear_mem_backend():
            for fn in fns:
                logger.info(f"Invoking function {fn.__name__}")
                self.get_func(fn)(args)

    @cache(backend=BackendFactory.actor.mem)
    def get_text_embedding(self, model: str):
        dir = GlobalStorage.get_instance().workdir / "embeddings" / slugify(model)
        return EmbeddingManager.from_disk(
            dir, HfModelArgs(embedding_model=model, customization="default")
        )

    @cache(backend=BackendFactory.actor.mem)
    def get_func(self, fncls: type[Fn]) -> Fn:
        return fncls(self)

    @cache(backend=BackendFactory.actor.mem)
    def get_func_name(self, fncls: type[Fn]) -> str:
        return fncls.__name__
