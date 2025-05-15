from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping, Optional, Sequence

from kgdata.models import Ontology
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from sm.dataset import Example, FullTable
from sm.namespaces.namespace import KnowledgeGraphNamespace
from sm.namespaces.prelude import KGName
from sm.outputs.semantic_model import ClassNode
from sm.typing import InternalID


@dataclass
class Schema:
    kgns: KnowledgeGraphNamespace
    classes: list[InternalID]
    class_index: dict[InternalID, int]
    class_labels: list[str]
    class_easy_labels: list[str]
    class_label_keys: dict[str, InternalID]
    props: list[InternalID]
    prop_index: dict[InternalID, int]
    prop_labels: list[str]
    prop_easy_labels: list[str]
    prop_label_keys: dict[str, InternalID]

    @cached_property
    def set_classes(self):
        return set(self.classes)

    @cached_property
    def set_props(self):
        return set(self.props)

    @staticmethod
    def from_ontology(
        ontology: Ontology,
        examples: Optional[Sequence[Example[FullTable]]] = None,
        used_props: Optional[Iterable[InternalID]] = None,
        used_classes: Optional[Iterable[InternalID]] = None,
    ) -> Schema:
        """Get the schema from the KGDB. If examples are provided, we will only include the classes and properties that are used in the examples.

        Args:
            kgdb: The KGDB object
            examples: The examples to use to filter the classes and properties. Defaults to None.
            used_props: The properties to include in the schema. Defaults to None.
            used_classes: The classes to include in the schema. Defaults to None.
        """
        if examples is not None:
            assert (
                used_props is None and used_classes is None
            ), "Cannot provide both examples and used_props/used_classes"
            used_props, used_classes = get_used_concepts(ontology.kgns, examples)

        if used_props is None:
            used_props = ontology.props.keys()

        if used_classes is None:
            used_classes = ontology.classes.keys()

        used_props = sorted(used_props)
        used_classes = sorted(used_classes)

        used_prop_labels = get_prop_easy_labels(
            ontology.kgname, ontology.props, used_props
        )
        used_class_labels = get_class_easy_labels(
            ontology.kgname, ontology.classes, used_classes
        )

        # name must be unique
        assert len(set(used_prop_labels)) == len(used_props)
        assert len(set(used_class_labels)) == len(used_classes)

        return Schema(
            kgns=ontology.kgns,
            props=used_props,
            prop_index={id: i for i, id in enumerate(used_props)},
            prop_labels=[str(ontology.props[id].label) for id in used_props],
            prop_easy_labels=used_prop_labels,
            prop_label_keys=get_label_keys(used_props, used_prop_labels),
            classes=used_classes,
            class_index={id: i for i, id in enumerate(used_classes)},
            class_labels=[str(ontology.classes[id].label) for id in used_classes],
            class_easy_labels=used_class_labels,
            class_label_keys=get_label_keys(used_classes, used_class_labels),
        )

    def get_class_label(self, id: InternalID) -> str:
        label = self.class_labels[self.class_index[id]]
        if self.kgns.has_encrypted_name(self.kgns.id_to_uri(id)):
            return f"{label} ({id})"
        return label

    def get_prop_label(self, id: InternalID) -> str:
        label = self.prop_labels[self.prop_index[id]]
        if self.kgns.has_encrypted_name(self.kgns.id_to_uri(id)):
            return f"{label} ({id})"
        return label


def get_prop_easy_labels(
    kgname: KGName, props: Mapping[InternalID, OntologyProperty], pids: list[InternalID]
) -> list[str]:
    if kgname == KGName.Wikidata:
        return [f"{str(props[id].label)} ({id})" for id in pids]
    return [f"{str(props[id].label)} (P{i})" for i, id in enumerate(pids)]


def get_class_easy_labels(
    kgname: KGName, classes: Mapping[InternalID, OntologyClass], cids: list[InternalID]
) -> list[str]:
    if kgname == KGName.Wikidata:
        return [f"{str(classes[id].label)} ({id})" for id in cids]
    return [f"{str(classes[id].label)} (Q{i})" for i, id in enumerate(cids)]


def get_label_keys(ids: list[InternalID], labels: list[str]) -> dict[str, InternalID]:
    """We assume that a label is generated to include a key (prop is P<number>, class is Q<number>) to make it easier to extract the results
    from LLM. This function creates a mapping from the label key to the id of the corresponding class/property in the list to make it easier to retrieve the original
    labels
    """
    index = {}
    for i, label in enumerate(labels):
        key = label.split(" ")[-1]
        assert re.match(r"\((Q|P)\d+\)", key) is not None, key
        index[key[1:-1]] = ids[i]
    return index


def get_used_concepts(
    kgns: KnowledgeGraphNamespace, examples: Sequence[Example[FullTable]]
) -> tuple[set[str], set[str]]:
    used_props = set()
    used_classes = set()

    for ex in examples:
        for sm in ex.sms:
            for edge in sm.iter_edges():
                if not kgns.is_uri_in_main_ns(edge.abs_uri):
                    continue
                pid = kgns.uri_to_id(edge.abs_uri)
                used_props.add(pid)

            for node in sm.iter_nodes():
                if not isinstance(node, ClassNode):
                    continue
                if not kgns.is_uri_in_main_ns(node.abs_uri):
                    continue
                cid = kgns.uri_to_id(node.abs_uri)
                used_classes.add(cid)
    return used_props, used_classes
