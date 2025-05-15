from __future__ import annotations

from typing import Sequence, TypedDict

import orjson
from gp.misc.itemdistance import KGItemDistance
from gpp.sem_label.feats._sample import GetSampleLabelOutput
from kgdata.models import Ontology, OntologyClass, OntologyProperty
from sm.dataset import Optional
from sm.misc.funcs import filter_duplication


class GetTargetLabelOutputItem(TypedDict):
    id: str
    is_prop: bool
    label: str
    aliases: list[str]
    description: str
    similar_labels: dict[str, list[int]]


GetTargetLabelOutput = Sequence[GetTargetLabelOutputItem]


def get_target_label(
    samples: GetSampleLabelOutput,
    ontology: Ontology,
    cls_distance: KGItemDistance,
    prop_distance: KGItemDistance,
    label_ids: Optional[Sequence[str]] = None,
) -> GetTargetLabelOutput:
    """Return a reference set of target labels containing information to be predicted."""
    if label_ids is None:
        label_ids = sorted(
            {x1 for x in samples["original_column_type"].value for x1 in x}
        )
    else:
        label_ids = sorted(label_ids)
    labels: list[OntologyClass | OntologyProperty] = [
        ontology.props[eid] if eid in ontology.props else ontology.classes[eid]
        for eid in label_ids
    ]

    return [
        {
            "id": label.id,
            "is_prop": isinstance(label, OntologyProperty),
            "label": str(label.label),
            "aliases": filter_duplication(
                [
                    value
                    for code in ["en", "en-gb", "en-us", "de", "fr", "es", "it"]
                    for value in label.aliases.lang2values.get(code, [])
                ]
            ),
            "description": str(label.description),
            "similar_labels": get_similar_labels(
                label, labels, cls_distance, prop_distance
            ),
        }
        for label in labels
    ]


def get_similar_labels(
    mainlbl: OntologyClass | OntologyProperty,
    labels: list[OntologyClass | OntologyProperty],
    cls_distance: KGItemDistance,
    prop_distance: KGItemDistance,
) -> dict[str, list[int]]:
    if isinstance(mainlbl, OntologyClass):
        fn_distance = cls_distance
    else:
        fn_distance = prop_distance

    sim_labels = {}

    for targetlbl in labels:
        if targetlbl.id == mainlbl.id:
            continue

        common_ancestors = set(mainlbl.ancestors.keys()).intersection(
            targetlbl.ancestors
        )
        common_ancestors_distance = [
            abs(dis)
            for dis in fn_distance.batch_get_distance(
                [
                    (cls.id, anc)
                    for cls in [mainlbl, targetlbl]
                    for anc in common_ancestors
                ]
            )
        ]
        if len(common_ancestors_distance) > 0 and all(
            dis <= 2 for dis in common_ancestors_distance
        ):
            if len(common_ancestors) > 0:
                assert type(targetlbl) is type(mainlbl)
            sim_labels[targetlbl.id] = min(common_ancestors_distance)
    return sim_labels
