from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Optional

from gp.actors.data import KGDB, KGDBArgs
from gp.misc.evaluation.sm_wikidata import SemanticModelHelper
from gp.semanticmodeling.postprocessing.cgraph import (
    CGEdge,
    CGEdgeTriple,
    CGNode,
    CGraph,
)
from gpp.actors.graph_space_actor import GraphSpace, SimpleEdge
from gpp.sem_model.from_sem_label._algo_base import (
    CandidateGraph,
    EstimatedScore,
    ISemModel,
    ScoreSource,
    SemType,
)
from gpp.sem_model.from_sem_label.steiner_tree_v100 import SteinerTreeV100
from gpp.sem_model.from_sem_label.steiner_tree_v101 import SteinerTreeV101
from kgdata.models import Ontology
from kgdata.models.ont_class import OntologyClass
from kgdata.models.ont_property import OntologyProperty
from libactor.cache import IdentObj
from rdflib import RDFS
from sm.dataset import Example, FullTable
from sm.inputs.table import ColumnBasedTable
from sm.misc.funcs import assert_not_null
from sm.misc.ray_helper import get_instance
from sm.outputs.semantic_model import SemanticModel
from sm.prelude import O

V200NP = dict[str, dict[str, float]]
V200EP = dict[CGEdgeTriple, dict[tuple[str, Optional[str]], dict[ScoreSource, float]]]


@dataclass
class V200CGProbs:
    node_probs: V200NP
    edge_probs: V200EP
    is_data_edge: dict[CGEdgeTriple, bool]


class AlgoV300(ISemModel[tuple[SemType, SemType], V200CGProbs]):
    def __init__(self, coeff_stscore: float, coeff_kgscore: float):
        self.coeff_stscore = coeff_stscore
        self.coeff_kgscore = coeff_kgscore

    def get_candidate_graph(
        self,
        ex: Example[FullTable],
        ex_stypes: dict[int, tuple[SemType, SemType]],
        ent_cols: set[int],
        ontology: Ontology,
        graphspace: GraphSpace,
    ) -> CandidateGraph[V200CGProbs]:
        """Get a candidate graph

        This algorithm generates a CPA graph instead of generating a full graph like in Mohsen's algorithm.

        Multiple same classes can be in the same graph.
        """
        data_props = graphspace.data_props
        object_props = graphspace.object_props

        cg = CGraph()
        node_probs: V200NP = defaultdict(dict)
        edge_probs: V200EP = defaultdict(lambda: defaultdict(dict))
        is_data_edge: dict[CGEdgeTriple, bool] = {}

        # indexes
        column2node = {}
        class2nodes = defaultdict(list)
        col2classes = defaultdict(set)
        id2statement = {}

        # first, we add column nodes
        for ci, (col_ctypes, col_ptypes) in ex_stypes.items():
            column2node[ci] = cg.add_node(
                CGNode(
                    id=f"col:{ci}",
                    is_statement_node=False,
                    is_column_node=True,
                    is_entity_node=False,
                    is_literal_node=False,
                    is_in_context=False,
                    column_index=ci,
                )
            )

        # then we add classes that can be associated with the columns
        for ci, (col_ctypes, col_ptypes) in ex_stypes.items():
            if ci in ent_cols:
                # we have entity columns
                for label, score in col_ctypes:
                    assert label not in ontology.props

                    class2nodes[label].append(column2node[ci])
                    col2classes[ci].add(label)
                    assert label not in node_probs[column2node[ci]]
                    node_probs[column2node[ci]][label] = score

        # now we can add relationships
        for ci, (col_ctypes, col_ptypes) in ex_stypes.items():
            for label, score in col_ptypes:
                assert label in ontology.props

                # queue containing pair of qual & prop that we need to check
                queue: list[SimpleEdge] = []
                queue.extend(data_props.get(None, {}).get(label, []))
                queue.extend(
                    sorted(
                        [
                            x
                            for prop, lst in data_props.get(label, {}).items()
                            for x in lst
                        ],
                        key=lambda x: x.inherit_freq,
                        reverse=True,
                    )
                )

                for canedge in queue:
                    qual = canedge.qual
                    prop = canedge.prop
                    assert qual is None or (qual != prop)
                    qual_or_prop = qual if qual is not None else prop

                    if canedge.source_type not in class2nodes:
                        # domain constraint is not satisfied
                        continue

                    if canedge.target_type is None:
                        if ci in ent_cols:
                            # we think this is an entity column
                            continue
                    else:
                        if ci not in ent_cols:
                            # we think this is a literal column
                            continue
                        if canedge.target_type not in col2classes[ci]:
                            # range constraint is not satisfied
                            continue

                    for uid in class2nodes[canedge.source_type]:
                        if uid == column2node[ci]:
                            continue

                        stmt_id = self.add_edge(
                            cg,
                            source_id=uid,
                            class_id=canedge.source_type,
                            target_id=column2node[ci],
                            prop=prop,
                            qual=qual_or_prop,
                            id2statement=id2statement,
                        )

                        cgtriple = (stmt_id, column2node[ci], qual_or_prop)
                        assert (
                            ScoreSource.FROM_STYPE_MODEL
                            not in edge_probs[cgtriple][
                                canedge.source_type, canedge.target_type
                            ]
                        )
                        edge_probs[cgtriple][canedge.source_type, canedge.target_type][
                            ScoreSource.FROM_STYPE_MODEL
                        ] = score
                        assert is_data_edge.get(
                            cgtriple, canedge.target_type is None
                        ) == (
                            canedge.target_type is None
                        ), f"To make sure that we don't have conflict values: {is_data_edge.get(cgtriple, canedge.target_type is None)} vs {canedge.target_type is None}"
                        is_data_edge[cgtriple] = canedge.target_type is None

        # how about relationships between class nodes?
        for source, snodes in class2nodes.items():
            for target, tnodes in class2nodes.items():
                if source == target:
                    continue

                pairs = [(uid, vid) for uid in snodes for vid in tnodes if uid != vid]
                for canedge in object_props.get((source, target), []):
                    assert canedge.qual is None or (canedge.qual != canedge.prop)
                    qual = canedge.qual if canedge.qual is not None else canedge.prop
                    for uid, vid in pairs:
                        stmt_id = self.add_edge(
                            cg,
                            source_id=uid,
                            class_id=source,
                            target_id=vid,
                            prop=canedge.prop,
                            qual=qual,
                            id2statement=id2statement,
                        )

                        cgtriple = (stmt_id, vid, qual)
                        assert (
                            ScoreSource.FROM_KG_MINING
                            not in edge_probs[cgtriple][source, target]
                        )
                        edge_probs[cgtriple][source, target][
                            ScoreSource.FROM_KG_MINING
                        ] = (canedge.inherit_freq / canedge.total_freq)
                        assert (
                            is_data_edge.get(cgtriple, False) == False
                        ), "To make sure that we don't have conflict values"
                        is_data_edge[cgtriple] = False

        return CandidateGraph(cg, V200CGProbs(node_probs, edge_probs, is_data_edge))

    def add_edge(
        self,
        cg: CGraph,
        source_id: str,
        class_id: str,
        target_id: str,
        prop: str,
        qual: str,
        id2statement: dict,
    ):
        # statement id is the combination of source id and property id
        # we can add class id into the combination, but this will results
        # in many duplicated edges
        stmt_id = (source_id, prop)

        if stmt_id not in id2statement:
            id2statement[stmt_id] = cg.add_node(
                CGNode(
                    id=f"stmt:{source_id}-{prop}",
                    is_statement_node=True,
                    is_column_node=False,
                    is_entity_node=False,
                    is_literal_node=False,
                    is_in_context=False,
                    column_index=None,
                )
            )
            cg.add_edge(
                CGEdge(
                    id=-1,
                    source=source_id,
                    target=id2statement[stmt_id],
                    key=prop,
                )
            )

        if not cg.has_edge_between_nodes(id2statement[stmt_id], target_id, qual):
            cg.add_edge(
                CGEdge(
                    id=-1,
                    source=id2statement[stmt_id],
                    target=target_id,
                    key=qual,
                )
            )
        return id2statement[stmt_id]

    def get_semantic_model(
        self,
        ex: Example[FullTable],
        cangraph: CandidateGraph[V200CGProbs],
        ontology: Ontology,
    ) -> SemanticModel:
        edge_probs = self.get_edge_probs(cangraph)

        st = SteinerTreeV100(ex.table, cangraph.cg, edge_probs, threshold=0.0)
        pred_cg = st.get_result()

        node_probs = self.get_node_probs(ex, cangraph, pred_cg)
        cta = {
            ci: label_score[0][0]
            for ci, label_score in node_probs.items()
            if len(label_score) > 0
        }
        # add back missing columns
        for u in cangraph.cg.iter_nodes():
            if u.column_index is not None:
                if u.column_index in cta and not pred_cg.has_node(u.id):
                    pred_cg.add_node(deepcopy(u))
        return self.to_semantic_model(ex.table.table, pred_cg, cta, ontology)

    def get_edge_probs(
        self, cangraph: CandidateGraph[V200CGProbs]
    ) -> dict[CGEdgeTriple, float]:
        edge_probs = {}
        for (sid, vid, edgekey), dict1 in cangraph.cg_probs.edge_probs.items():
            assert cangraph.cg.get_node(sid).is_statement_node
            (usedge,) = cangraph.cg.in_edges(sid)
            uprobs = cangraph.cg_probs.node_probs[usedge.source]
            vprobs = cangraph.cg_probs.node_probs[vid]

            all_scores = []
            if cangraph.cg_probs.is_data_edge[sid, vid, edgekey]:
                # data edge
                for (source_cls, target_cls), dict2 in dict1.items():
                    assert len(dict2) == 1
                    assert target_cls is None
                    score = dict2[ScoreSource.FROM_STYPE_MODEL] * uprobs[source_cls]
                    all_scores.append(score)
            else:
                # object props
                for (source_cls, target_cls), dict2 in dict1.items():
                    assert target_cls is not None
                    score = 0.0
                    if ScoreSource.FROM_STYPE_MODEL in dict2:
                        score += (
                            dict2[ScoreSource.FROM_STYPE_MODEL]
                            * uprobs[source_cls]
                            * vprobs[target_cls]
                        ) * self.coeff_stscore
                    elif ScoreSource.FROM_KG_MINING in dict2:
                        score += (
                            dict2[ScoreSource.FROM_KG_MINING]
                            * uprobs[source_cls]
                            * vprobs[target_cls]
                        ) * self.coeff_kgscore
                    all_scores.append(score)

            assert len(all_scores) > 0
            edge_score = max(all_scores)
            edge_probs[sid, vid, edgekey] = edge_score

        # the incoming edge of the statement is going to be the maximum probability
        # of each edge.
        for usedge in cangraph.cg.iter_edges():
            snode = cangraph.cg.get_node(usedge.target)
            if not snode.is_statement_node:
                continue

            ustriple = (usedge.source, usedge.target, usedge.key)
            assert ustriple not in edge_probs
            edge_probs[ustriple] = max(
                edge_probs[e.source, e.target, e.key]
                for e in cangraph.cg.out_edges(usedge.target)
            )
        return edge_probs

    def get_node_probs(
        self,
        ex: Example[FullTable],
        cangraph: CandidateGraph[V200CGProbs],
        filtered_cg: CGraph,
    ) -> dict[int, list[tuple[str, float]]]:
        return {
            assert_not_null(cangraph.cg.get_node(k).column_index): sorted(
                v.items(), key=lambda x: x[1], reverse=True
            )
            for k, v in cangraph.cg_probs.node_probs.items()
        }

    # def get_node_probs(
    #     self,
    #     ex: Example[FullTable],
    #     cangraph: CandidateGraph[V200CGProbs],
    #     filtered_cg: CGraph,
    # ) -> dict[int, list[tuple[str, float]]]:
    #     new_node_probs = {}

    #     # the given node probs only contains the probability of the class that is associated with
    #     # so we can use it.
    #     for uid, label2score in cangraph.cg_probs.node_probs.items():
    #         u = filtered_cg.get_node(uid)
    #         col_index = u.column_index
    #         assert col_index is not None

    #         outedges = filtered_cg.out_edges(uid)
    #         source2score = defaultdict(float)
    #         for usedge in filtered_cg.out_edges(uid):
    #             for svedge in filtered_cg.out_edges(usedge.target):
    #                 for (source, target), item2score in cangraph.cg_probs.edge_probs[
    #                     svedge.source, svedge.target, svedge.key
    #                 ].items():
    #                     source2score[source] += (
    #                         item2score.get(ScoreSource.FROM_STYPE_MODEL, 0.0)
    #                         * self.coeff_stscore
    #                         + item2score.get(ScoreSource.FROM_KG_MINING, 0.0)
    #                         * self.coeff_kgscore
    #                     ) * cangraph.cg_probs.node_probs[uid][source]
    #         new_node_probs[col_index] = sorted(
    #             source2score.items(), key=lambda x: x[1], reverse=True
    #         )
    #     print(new_node_probs)
    #     return new_node_probs

    def to_semantic_model(
        self,
        table: ColumnBasedTable,
        cg: CGraph,
        cta: Mapping[int, str],
        ontology: Ontology,
    ) -> SemanticModel:
        """Convert a candidate graph (CPA) to a semantic model"""
        nodes = {
            u.id: u.to_sm_node(
                table, ontology.kgns, id2literal=lambda x: x.split(":", 1)[1]
            )
            for u in cg.iter_nodes()
        }
        cpa = [(edge.source, edge.target, edge.key) for edge in cg.iter_edges()]
        return SemanticModelHelper.from_ontology(ontology).create_sm(
            nodes, cpa, cta, on_untype_source_column_node="create-class"
        )


class AlgoV301(AlgoV300):
    def get_semantic_model(
        self,
        ex: Example[FullTable],
        cangraph: CandidateGraph[V200CGProbs],
        ontology: Ontology,
    ) -> SemanticModel:
        edge_probs = self.get_edge_probs(cangraph)

        st = SteinerTreeV101(ex.table, cangraph.cg, edge_probs, threshold=0.0)
        pred_cg = st.get_result()

        node_probs = self.get_node_probs(ex, cangraph, pred_cg)
        cta = {
            ci: label_score[0][0]
            for ci, label_score in node_probs.items()
            if len(label_score) > 0
        }
        # add back missing columns
        for u in cangraph.cg.iter_nodes():
            if u.column_index is not None:
                if u.column_index in cta and not pred_cg.has_node(u.id):
                    pred_cg.add_node(deepcopy(u))
        return self.to_semantic_model(ex.table.table, pred_cg, cta, ontology)

    def get_edge_probs(
        self, cangraph: CandidateGraph[V200CGProbs]
    ) -> dict[CGEdgeTriple, float]:
        edge_probs = {}
        for (sid, vid, edgekey), dict1 in cangraph.cg_probs.edge_probs.items():
            assert cangraph.cg.get_node(sid).is_statement_node
            (usedge,) = cangraph.cg.in_edges(sid)
            uprobs = cangraph.cg_probs.node_probs[usedge.source]
            vprobs = cangraph.cg_probs.node_probs[vid]

            all_scores = []
            if cangraph.cg_probs.is_data_edge[sid, vid, edgekey]:
                # data edge
                for (source_cls, target_cls), dict2 in dict1.items():
                    assert len(dict2) == 1
                    assert target_cls is None
                    score = dict2[ScoreSource.FROM_STYPE_MODEL] * uprobs[source_cls]
                    all_scores.append(score)
            else:
                # object props
                for (source_cls, target_cls), dict2 in dict1.items():
                    assert target_cls is not None
                    score = 0.0
                    if ScoreSource.FROM_STYPE_MODEL in dict2:
                        score += (
                            dict2[ScoreSource.FROM_STYPE_MODEL]
                            * uprobs[source_cls]
                            * vprobs[target_cls]
                        ) * self.coeff_stscore
                    if ScoreSource.FROM_KG_MINING in dict2:
                        score += (
                            dict2[ScoreSource.FROM_KG_MINING]
                            * uprobs[source_cls]
                            * vprobs[target_cls]
                        ) * self.coeff_kgscore
                    all_scores.append(score)

            assert len(all_scores) > 0
            edge_score = max(all_scores)
            edge_probs[sid, vid, edgekey] = edge_score

        # the incoming edge of the statement is going to be the maximum probability
        # of each edge.
        for usedge in cangraph.cg.iter_edges():
            snode = cangraph.cg.get_node(usedge.target)
            if not snode.is_statement_node:
                continue

            ustriple = (usedge.source, usedge.target, usedge.key)
            assert ustriple not in edge_probs
            edge_probs[ustriple] = max(
                edge_probs[e.source, e.target, e.key]
                for e in cangraph.cg.out_edges(usedge.target)
            )
        return edge_probs
