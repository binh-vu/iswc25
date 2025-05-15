from __future__ import annotations

from typing import Sequence, TypedDict

from gpp.sem_label.feats._example import (
    EntityWithScore,
    GetExamplesOutput,
    LiteralWithScore,
)
from gpp.sem_label.feats._target_label import GetTargetLabelOutput
from kgdata.models import Ontology
from kgdata.wikidata.models.wdvalue import WDValue
from rdflib import Literal


class GetTargetLabelExampleOutputItem(TypedDict):
    id: str
    examples: Sequence[str]


GetTargetLabelExampleOutput = Sequence[GetTargetLabelExampleOutputItem]


def get_target_label_examples(
    target_labels: GetTargetLabelOutput,
    target_label_examples: GetExamplesOutput,
    n_examples_per_label: int = 100,
):
    """
    Retrieves target label examples from the knowledge graph.

    Args:
        target_labels: Target labels.
        ontology: The ontology to use for retrieving examples.
        n_examples_per_label: Number of examples to retrieve per label.
    """
    output = []
    for label, lbl_examples in zip(
        target_labels,
        target_label_examples,
    ):
        # assert len(lbl_examples) > 0, f"Label {label['id']} has no examples"

        # convert examples into texts to add to embeddings
        examples = []
        for lbl_ex in lbl_examples[:n_examples_per_label]:
            if isinstance(lbl_ex, EntityWithScore):
                text = str(lbl_ex.label)
            else:
                assert isinstance(lbl_ex, LiteralWithScore)
                value = lbl_ex.value
                if isinstance(value, Literal):
                    text = str(value)
                else:
                    assert isinstance(value, WDValue)
                    # handle wdvalue here.
                    if WDValue.is_string(value):
                        text = value.as_string()
                    elif WDValue.is_quantity(value):
                        text = str(value.value["amount"])
                    elif WDValue.is_mono_lingual_text(value):
                        text = str(value.value["text"])
                    elif WDValue.is_globe_coordinate(value):
                        text = format_globe_coordinate(
                            value.value["latitude"], value.value["longitude"]
                        )
                    else:
                        assert WDValue.is_time(value), value
                        text = value.value["time"]

            examples.append(text)

        output.append({"id": label["id"], "example": examples})

    return output


def format_globe_coordinate(lat: float, long: float):
    """Format globe coordinate in N, S, E, W"""
    # https://www.wikihow.com/Write-Latitude-and-Longitude
    if lat >= 0:
        slat = f"{lat:.3f}°E"
    else:
        slat = f"{-lat:.3f}°W"

    if long >= 0:
        slong = f"{long:.3f}°N"
    else:
        slong = f"{-long:.3f}°S"

    return f"{slong} {slat}"
