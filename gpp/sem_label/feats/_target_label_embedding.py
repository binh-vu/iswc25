from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Sequence, TypedDict, Union

import numpy as np
import orjson
from gpp.sem_label.feats._sample import GetSampleLabelOutput
from gpp.sem_label.feats._target_label import GetTargetLabelOutput
from keyvec import EmbeddingManager
from kgdata.misc.ntriples_parser import node_from_dict, node_to_dict
from kgdata.models import (
    MultiLingualString,
    MultiLingualStringList,
    Ontology,
    OntologyClass,
    OntologyProperty,
)
from kgdata.wikidata.models import WDValueKind
from kgdata.wikidata.models.wdvalue import WDValue, WDValueKind
from rdflib import Literal, URIRef
from sm.namespaces.namespace import KnowledgeGraphNamespace
from smml.data_model_helper import NP2DArray


class GetTargetLabelEmbeddingOutput(TypedDict):
    id2index: dict[str, int]
    embeddings: Annotated[np.ndarray, "NDArray 2D (n_labels, emb_dim)"]
    labels: Sequence[OntologyClass | OntologyProperty]


def get_target_label_embedding(
    sample_labels: GetSampleLabelOutput, ontology: Ontology, textemb: EmbeddingManager
) -> GetTargetLabelEmbeddingOutput:
    """Return a reference set of target labels that we want to predict. This reference set allows
    us to convert a class/property ID in string to an integer number."""

    label_ids = sorted(
        {x1 for x in sample_labels["original_column_type"].value for x1 in x}
    )
    labels: list[OntologyClass | OntologyProperty] = [
        ontology.props[eid] if eid in ontology.props else ontology.classes[eid]
        for eid in label_ids
    ]

    # get class embeddings
    label_emb = textemb.batch_get(
        [
            str(
                ontology.props[eid].label
                if eid in ontology.props
                else ontology.classes[eid].label
            )
            for eid in label_ids
        ]
    )
    id2index = {eid: i for i, eid in enumerate(label_ids)}

    return GetTargetLabelEmbeddingOutput(
        id2index=id2index,
        embeddings=label_emb,
        labels=labels,
    )
