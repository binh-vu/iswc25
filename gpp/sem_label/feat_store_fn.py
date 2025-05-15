from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_type_hints

from gp.misc.appconfig import AppConfig
from libactor.misc import assign_dataclass_field_names
from ream.cache_helper import CacheableFn, MemBackend

if TYPE_CHECKING:
    from gpp.sem_label.feat_store import STypeStore


@dataclass
class STypeStoreFnArgs:
    n_neg_labels: int = field(
        metadata={"help": "The number of negative labels for training"}
    )
    neg_sample_spaces: Literal["from_dataset"] = field(
        metadata={"help": "The number of negative sample spaces"}
    )
    min_freq_over_row: float = field(
        metadata={
            "help": "Minimum frequency over row that is used to filter out unlikely correct edges"
        },
    )
    max_unmatch_over_ent_row: float = field(
        metadata={
            "help": "Maximum frequency of unmatched entities over row that is used to filter out unlikely correct edges"
        },
    )
    text_embedding_model: str = field(
        metadata={"help": "The embedding model to calculate text embeddings"},
    )
    n_examples_per_column: int = field(
        default=100, metadata={"help": "The number of examples per column"}
    )
    n_examples_per_label: int = field(
        default=100, metadata={"help": "The number of examples per label"}
    )


assign_dataclass_field_names(STypeStoreFnArgs)


class STypeStoreFn(CacheableFn):
    backends = []
    use_args = []

    def __init__(self, store: STypeStore):
        super().__init__(
            self.use_args,
            store.get_working_fs(),
            not AppConfig.get_instance().is_cache_enable,
        )
        # to assign functions based on type hints
        for name, type in get_type_hints(self.__class__).items():
            if issubclass(type, CacheableFn):
                setattr(self, name, store.get_func(type))
        self.store = store

    @staticmethod
    def cache_decorator_args():
        return {
            "cache_key": CacheableFn.get_cache_key,
            "disable": "disable",
        }

    def get_diskname(self, args: STypeStoreFnArgs):
        if hasattr(self, "VERSION"):
            version = f"{getattr(self, 'VERSION')}/"
        else:
            version = ""

        name = f"%s/{version}%s" % (
            re.sub(
                r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__.replace("Fn", "")
            ).lower(),
            args.dsquery,
        )
        return name

    @classmethod
    def new_mem_backend(cls):
        backend = MemBackend()
        cls.backends.append(backend)
        return backend

    @classmethod
    @contextmanager
    def auto_clear_mem_backend(cls):
        try:
            yield None
        finally:
            for backend in cls.backends:
                backend.clear()
