from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
from gpp.llm.qa_llm import Schema
from kgdata.models import Ontology
from kgdata.wikidata.datasets.meta_graph_stats import PConnection
from libactor.actor import Actor
from libactor.cache import BackendFactory, IdentObj, cache
from sm.dataset import Example, FullTable
from sm.misc.funcs import assert_one_item, group_by
from sm.namespaces.prelude import KnowledgeGraphNamespace
from sm.outputs.semantic_model import ClassNode, DataNode, LiteralNode


@dataclass
class SimpleEdge:
    prop: str
    qual: Optional[str]
    source_type: str
    target_type: Optional[str]
    freq: float
    inherit_freq: float
    total_freq: float


# mapping from qual -> prop -> list of edges
DataPropSchema = dict[Optional[str], dict[str, list[SimpleEdge]]]
# mapping from (source_type, target_type) -> list of edges
ObjectPropSchema = dict[tuple[str, str], list[SimpleEdge]]


@dataclass
class GraphSpace:
    data_props: DataPropSchema
    object_props: ObjectPropSchema

    def __repr__(self):
        return "GraphSpace(data_props=..., object_props=...)"

    def as_df(self, ontology: Ontology):
        return pd.DataFrame(
            [
                {
                    "prop": ontology.get_prop_label(prop),
                    "qual": ontology.get_prop_label(qual) if qual is not None else None,
                    "source_type": ontology.get_class_label(item.source_type),
                    "target_type": (
                        ontology.get_class_label(item.target_type)
                        if item.target_type is not None
                        else None
                    ),
                    "freq": item.freq,
                    "inherit_freq": item.inherit_freq,
                    "total_freq": item.total_freq,
                }
                for qual in self.data_props
                for prop, lst in self.data_props[qual].items()
                for item in lst
            ]
        ), pd.DataFrame(
            [
                {
                    "prop": ontology.get_prop_label(item.prop),
                    "qual": (
                        ontology.get_prop_label(item.qual)
                        if item.qual is not None
                        else None
                    ),
                    "source_type": ontology.get_class_label(source_type),
                    "target_type": ontology.get_class_label(target_type),
                    "freq": item.freq,
                    "inherit_freq": item.inherit_freq,
                    "total_freq": item.total_freq,
                }
                for (source_type, target_type), lst in self.object_props.items()
                for item in lst
            ]
        )


@dataclass
class GraphSpaceV1Args:
    top_k_data_props: Optional[int] = None
    top_k_object_props: Optional[int] = None


class GraphSpaceV1Actor(Actor[GraphSpaceV1Args]):
    VERSION = 103

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(
            mem_persist=True, log_serde_time=True
        )
    )
    def forward(
        self,
        train_examples: Optional[IdentObj[Sequence[Example[FullTable]]]],
        schema: IdentObj[Schema],
        ontology: IdentObj[Ontology],
        predicate_connections: Optional[IdentObj[list[PConnection]]] = None,
    ) -> IdentObj[GraphSpace]:
        """Get data props as well as object props"""
        used_props = schema.value.props
        used_classes = schema.value.classes
        if predicate_connections is None:
            # figure out the predicate_connections from the training examples
            assert train_examples is not None
            obj: IdentObj[list[PConnection]] = (
                self.get_predicate_connections_from_examples(
                    train_examples, ontology.value.kgns
                )
            )
            predicate_connections_key = obj.key
            predicate_connections_value = obj.value
        else:
            predicate_connections_key = predicate_connections.key
            predicate_connections_value: list[PConnection] = predicate_connections.value

        data_props = {
            qual: group_by(lst, lambda x: x.prop)
            for qual, lst in group_by(
                predicate_connections_value, lambda x: x.qual
            ).items()
        }
        object_props = group_by(
            predicate_connections_value, lambda x: (x.source_type, x.target_type)
        )

        new_data_props: dict[Optional[str], dict[str, list[SimpleEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        new_object_props: dict[tuple[str, str], list[SimpleEdge]] = {}

        for qual in list(used_props) + [None]:
            if qual not in data_props:
                continue

            for prop in used_props:
                if prop not in data_props[qual]:
                    continue

                # combination of (source & target types) that can be found in the dataset
                lst = [
                    conn
                    for conn in data_props[qual][prop]
                    if conn.source_type in used_classes
                    and (conn.target_type is None or conn.target_type in used_classes)
                ]

                newlst: list[SimpleEdge] = []
                # to count the frequency of each (source & target types) combination with inheritance, we need
                # to search their ancestors
                descendants = {
                    source: {
                        target: assert_one_item(tmp1)
                        for target, tmp1 in group_by(
                            tmp, lambda x: x.target_type
                        ).items()
                    }
                    for source, tmp in group_by(lst, lambda x: x.source_type).items()
                }
                ancestors = {
                    source: {
                        target: assert_one_item(tmp1)
                        for target, tmp1 in group_by(
                            tmp, lambda x: x.target_type
                        ).items()
                    }
                    for source, tmp in group_by(
                        data_props[qual][prop], lambda x: x.source_type
                    ).items()
                }

                for des_sourceid, des_targets in descendants.items():
                    des_source = ontology.value.classes[des_sourceid]

                    inheriting_ancestors = {
                        anc_sourceid: anc_targets
                        for anc_sourceid, anc_targets in ancestors.items()
                        if anc_sourceid in des_source.ancestors
                    }
                    for des_targetid, des_conn in des_targets.items():
                        if des_targetid is not None:
                            des_target = ontology.value.classes[des_targetid]
                        else:
                            des_target = None

                        newrecord = SimpleEdge(
                            prop=prop,
                            qual=qual,
                            source_type=des_sourceid,
                            target_type=des_targetid,
                            freq=des_conn.freq,
                            inherit_freq=des_conn.freq,
                            total_freq=-1,
                        )

                        for anc_sourceid, anc_targets in inheriting_ancestors.items():
                            for anc_targetid, anc_conn in anc_targets.items():
                                if des_targetid is None:
                                    if anc_targetid is None:
                                        newrecord.inherit_freq += anc_conn.freq
                                else:
                                    assert des_target is not None
                                    if anc_targetid in des_target.ancestors:
                                        newrecord.inherit_freq += anc_conn.freq

                        newlst.append(newrecord)

                if len(newlst) > 0:
                    new_data_props[qual][prop] = sorted(
                        newlst, key=lambda x: x.inherit_freq, reverse=True
                    )

            # update total frequency -- this is to normalize
            # the likelihood of choosing source given a prop.
            total_freq = sum(
                edge.inherit_freq
                for edges in new_data_props[qual].values()
                for edge in edges
            )
            for edges in new_data_props[qual].values():
                for edge in edges:
                    edge.total_freq = total_freq

        for (source_type, target_type), conns in object_props.items():
            if (
                target_type is None
                or source_type not in used_classes
                or target_type not in used_classes
            ):
                continue

            conns = [
                conn
                for conn in conns
                if conn.prop in used_props
                and (conn.qual is None or conn.qual in used_props)
            ]

            total_freq = sum(conn.freq for conn in conns)
            new_object_props[source_type, target_type] = sorted(
                [
                    SimpleEdge(
                        conn.prop,
                        conn.qual,
                        conn.source_type,
                        conn.target_type,
                        conn.freq,
                        conn.freq,
                        total_freq,
                    )
                    for conn in conns
                ],
                key=lambda x: x.inherit_freq,
                reverse=True,
            )

        if self.params.top_k_data_props is not None:
            new_data_props = {
                qual: {
                    prop: lst1[: self.params.top_k_data_props]
                    for prop, lst1 in dict1.items()
                }
                for qual, dict1 in new_data_props.items()
            }
        if self.params.top_k_object_props is not None:
            new_object_props = {
                k: lst1[: self.params.top_k_object_props]
                for k, lst1 in new_object_props.items()
            }

        key = f"{self.key}[exs={train_examples.key if train_examples is not None else ""},schema={schema.key},ontology={ontology.key},pconns={predicate_connections_key}]"
        return IdentObj(
            key=key,
            value=GraphSpace(new_data_props, new_object_props),
        )

    @cache(
        backend=BackendFactory.actor.sqlite.pickle(
            mem_persist=True, log_serde_time=True
        ),
        cache_args=["train_examples"],
    )
    def get_predicate_connections_from_examples(
        self,
        train_examples: IdentObj[Sequence[Example[FullTable]]],
        kgns: KnowledgeGraphNamespace,
    ) -> IdentObj[list[PConnection]]:
        """Get predicate connections from the training examples"""
        sms = [sm for ex in train_examples.value for sm in ex.sms]
        key2count = defaultdict(int)

        for sm in sms:
            for edge in sm.iter_edges():
                edge_prop_id = kgns.uri_to_id(edge.abs_uri)

                u = sm.get_node(edge.source)
                v = sm.get_node(edge.target)
                if isinstance(v, ClassNode) and v.abs_uri == kgns.statement_uri:
                    # this is a statement node, skip as we do not want to count twice
                    continue

                if isinstance(v, ClassNode):
                    target_type_id = kgns.uri_to_id(v.abs_uri)
                else:
                    assert isinstance(v, (DataNode, LiteralNode))
                    # for literal nodes, we do not have the entity database so we can't access to
                    # its type information
                    target_type_id = None

                if isinstance(u, ClassNode):
                    if u.abs_uri == kgns.statement_uri:
                        u_inedge = sm.in_edges(u.id)[0]
                        u_parent = sm.get_node(u_inedge.source)
                        if isinstance(u_parent, ClassNode):
                            source_type_id = kgns.uri_to_id(u_parent.abs_uri)
                        elif isinstance(u_parent, LiteralNode):
                            # skip literal nodes because we do not provide the entity database
                            # so we can't access to its type
                            continue
                        else:
                            raise ValueError(
                                f"Unreachable code, cannot have outgoing edge from data node"
                            )

                        prop = kgns.uri_to_id(u_inedge.abs_uri)
                        qual = None if edge_prop_id == prop else edge_prop_id
                    else:
                        source_type_id = kgns.uri_to_id(u.abs_uri)
                        prop = edge_prop_id
                        qual = None
                elif isinstance(u, LiteralNode):
                    # skip literal nodes because we do not provide the entity database
                    # so we can't access to its type
                    continue
                else:
                    raise ValueError(
                        f"Unreachable code, cannot have outgoing edge from data node"
                    )

                key = (prop, qual, source_type_id, target_type_id)
                key2count[key] += 1

        return IdentObj(
            key=train_examples.key,
            value=[
                PConnection(
                    prop=prop,
                    qual=qual,
                    source_type=source_type,
                    target_type=target_type,
                    freq=freq,
                )
                for (prop, qual, source_type, target_type), freq in key2count.items()
            ],
        )

    # @cache(
    #     backend=BackendFactory.actor.sqlite.pickle(
    #         mem_persist=True, log_serde_time=True
    #     ),
    # )
    # def get_predicate_connections_from_ontology(self, ontology: IdentObj[Ontology]):
    #     """Get predicate connections from the ontology"""
    #     ont = ontology.value
    #     output = []

    #     for prop in ont.props.values():
    #         if len(prop.domains) == 0:
    #             continue
    #         for domain in prop.domains:
    #             if len(prop.ranges) == 0:
    #                 output.append(PConnection(prop.id, None, domain, None, 1))
    #             else:
    #                 for range in prop.ranges:
    #                     output.append(PConnection(prop.id, None, domain, range, 1))

    #     return IdentObj(
    #         key=ontology.key,
    #         value=output,
    #     )
