from __future__ import annotations

from typing import Optional, Sequence, TypedDict

import numpy as np
import sm.outputs as O
from gpp.config import IDENT_PROPS
from sm.dataset import Example, FullTable
from sm.misc.prelude import IntegerEncoder
from sm.namespaces.prelude import KnowledgeGraphNamespace
from smml.data_model_helper import SingleNumpyArray


class AutoLabelSemanticModel(O.SemanticModel):
    def __init__(self, check_cycle=False, multigraph=True):
        super().__init__(check_cycle=check_cycle, multigraph=multigraph)
        # mapping from edge id to its probability
        self.edge_probs: dict[int, float] = {}


class GetSampleLabelOutput(TypedDict):
    sample_id: SingleNumpyArray
    table_id: SingleNumpyArray
    column_index: SingleNumpyArray
    column_name: SingleNumpyArray
    column_type: SingleNumpyArray
    original_column_type: SingleNumpyArray
    probable_column_types: SingleNumpyArray
    original_probable_column_types: SingleNumpyArray


def get_sample_label(
    exs: Sequence[Example[FullTable]],
    kgns: KnowledgeGraphNamespace,
    ignore_no_type_column: bool = True,
) -> GetSampleLabelOutput:
    sample_id = []
    table_id = []
    column_index = []
    column_name = []
    column_type = IntegerEncoder()
    probable_column_types = IntegerEncoder()

    for ex in exs:
        for col in ex.table.table.columns:
            label = set()
            probable_label = set()

            for sm in ex.sms:
                if not sm.has_data_node(col.index):
                    continue

                u = sm.get_data_node(col.index)

                if isinstance(sm, AutoLabelSemanticModel):
                    # if the semantic model is an auto-label model, we do not trust all of its labels
                    # rather, we want to pick the most frequent label as the ground-truth
                    cta = {}
                    cpa = {}
                    for inedge in sm.in_edges(u.id):
                        pu = sm.get_node(inedge.source)
                        if inedge.abs_uri in IDENT_PROPS:
                            assert (
                                isinstance(pu, O.ClassNode)
                                and pu.abs_uri != kgns.statement_uri
                            ), "not support statement in auto-label semantic model"
                            cta[pu.abs_uri] = sm.edge_probs[inedge.id]
                            for inedge2 in sm.in_edges(inedge.source):
                                assert inedge2.abs_uri not in IDENT_PROPS
                                cpa[inedge2.abs_uri] = sm.edge_probs[inedge2.id]
                        else:
                            cpa[inedge.abs_uri] = sm.edge_probs[inedge.id]

                    # pick the most frequent label for each cta & cpa as the label ground-truth
                    # the rest goes to probable_column_types
                    if len(cta) > 0:
                        best_cta = max(cta.items(), key=lambda x: x[1])
                        label.add(best_cta[0])
                        probable_label.update(
                            [k for k, v in cta.items() if v != best_cta[0]]
                        )
                    if len(cpa) > 0:
                        best_cpa = max(cpa.items(), key=lambda x: x[1])
                        label.add(best_cpa[0])
                        probable_label.update(
                            [k for k, v in cpa.items() if v != best_cpa[0]]
                        )
                else:
                    for t in sm.get_semantic_types_of_column(col.index):
                        if t.predicate_abs_uri in IDENT_PROPS:
                            label.add(t.class_abs_uri)

                            # also add the incoming property/qualifier
                            # note that usually a column has only one class, however; for autolabel
                            # dataset, we will have because we want to keep track of multiple possible high freq classes
                            # so that we do not accidentally adding them into negative samples.
                            # Also, this loop is quite inefficient, but we do not encounter this case often and label
                            # is a set, so it will not be duplicated.
                            for inedge in sm.in_edges(u.id):
                                for inedge2 in sm.in_edges(inedge.source):
                                    assert inedge2.abs_uri not in IDENT_PROPS
                                    label.add(inedge2.abs_uri)
                        elif t.qualifier_abs_uri is None:
                            label.add(t.predicate_abs_uri)
                        else:
                            label.add(t.qualifier_abs_uri)

            if ignore_no_type_column and len(label) == 0:
                continue

            sample_id.append(len(sample_id))
            table_id.append(ex.table.table.table_id)
            column_index.append(col.index)
            column_name.append(col.clean_multiline_name)
            column_type.append(tuple(sorted((kgns.uri_to_id(uri) for uri in label))))
            probable_column_types.append(
                tuple(sorted((kgns.uri_to_id(uri) for uri in probable_label)))
            )

    # can't use np.array(column_type.get_decoder(), dtype=np.object_) if all elements are one item list
    # as it will create a 2D array.
    encoded_column_type = column_type.get_decoder()
    original_column_type = np.empty((len(encoded_column_type),), dtype=np.object_)
    original_column_type[:] = encoded_column_type

    encoded_probable_column_types = probable_column_types.get_decoder()
    original_probable_column_types = np.empty(
        (len(encoded_probable_column_types),), dtype=np.object_
    )
    original_probable_column_types[:] = encoded_probable_column_types

    return {
        "sample_id": SingleNumpyArray(np.array(sample_id)),
        "table_id": SingleNumpyArray(np.array(table_id, dtype=np.object_)),
        "column_index": SingleNumpyArray(np.array(column_index)),
        "column_name": SingleNumpyArray(np.array(column_name, dtype=np.object_)),
        "column_type": SingleNumpyArray(np.array(column_type.values)),
        "original_column_type": SingleNumpyArray(original_column_type),
        "original_probable_column_types": SingleNumpyArray(
            original_probable_column_types
        ),
        "probable_column_types": SingleNumpyArray(
            np.array(probable_column_types.values)
        ),
    }
