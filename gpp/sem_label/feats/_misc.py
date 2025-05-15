from __future__ import annotations

import os
from typing import Union

from gp.misc.itemdistance import KGItemDistance
from keyvec import EmbeddingManager, HfModel, HfModelArgs
from kgdata.models import Ontology
from libactor.cache import BackendFactory, IdentObj, cache
from libactor.storage import GlobalStorage
from loguru import logger
from slugify import slugify
from sm.misc.ray_helper import RemoteClient


@cache(backend=BackendFactory.func.mem)
def get_class_distance(ontology: IdentObj[Ontology]) -> IdentObj[KGItemDistance]:
    ont = ontology.value
    return IdentObj(
        key=f"{ontology.key}:class_distance",
        value=KGItemDistance(ont.classes, ont.kgns),
    )


@cache(backend=BackendFactory.func.mem)
def get_property_distance(ontology: IdentObj[Ontology]) -> IdentObj[KGItemDistance]:
    ont = ontology.value
    return IdentObj(
        key=f"{ontology.key}:property_distance",
        value=KGItemDistance(ont.props, ont.kgns),
    )


@cache(backend=BackendFactory.func.mem)
def get_text_embedding(model: str, customization: str = "default") -> EmbeddingManager:
    dir = GlobalStorage.get_instance().workdir / "embeddings" / slugify(model)
    if os.environ.get("HF_REMOTE") is not None:
        logger.info(
            "Using a remote server to run HuggingFace model. The server must be running and accessible through localhost port forwarding"
        )
        hfmodel: HfModelArgs = RemoteClient(
            HfModel,
            (HfModelArgs(embedding_model=model, customization=customization),),
            os.environ["HF_REMOTE"],
        )  # type: ignore
    else:
        hfmodel = HfModelArgs(embedding_model=model, customization=customization)

    return EmbeddingManager.from_disk(dir, hfmodel)
