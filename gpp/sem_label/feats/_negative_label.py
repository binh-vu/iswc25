from __future__ import annotations

from operator import itemgetter
from typing import TypedDict

import numpy as np
from gp.misc.itemdistance import KGItemDistance
from gpp.sem_label.feats._sample import GetSampleLabelOutput
from gpp.sem_label.feats._target_label_embedding import GetTargetLabelEmbeddingOutput
from kgdata.models import OntologyClass
from smml.data_model_helper import Single2DNumpyArray
from smml.dataset import EmbeddingFeat


class GetNegativeLabelOutput(TypedDict):
    pos_labels: EmbeddingFeat
    neg_labels: EmbeddingFeat
    pos_labels_mask: Single2DNumpyArray
    neg_labels_mask: Single2DNumpyArray


def get_negative_label(
    examples: GetSampleLabelOutput,
    target_label_emb: GetTargetLabelEmbeddingOutput,
    n_neg_labels: int,
    cls_distance: KGItemDistance,
    prop_distance: KGItemDistance,
):
    """For each example, generate positive labels (we may have more than one positive labels -- original_column_type)
    and corresponding negative labels. To make it have equal length, we pad the missing values with -1 and create a mask
    for vectorization later.

    Args:
        examples: example labels
        target_label_emb: target label embeddings
        n_neg_labels: number of negative labels to generate for training
        cls_distance: class distance
        prop_distance: property distance
    """
    lblembs = target_label_emb["embeddings"]
    original_output = get_negative_label_real(
        examples, target_label_emb, n_neg_labels, cls_distance, prop_distance
    )
    return GetNegativeLabelOutput(
        pos_labels=EmbeddingFeat(original_output["pos_labels"].value, lblembs),
        neg_labels=EmbeddingFeat(original_output["neg_labels"].value, lblembs),
        pos_labels_mask=Single2DNumpyArray(
            (original_output["pos_labels"].value != -1).astype(np.float32)
        ),
        neg_labels_mask=Single2DNumpyArray(
            (original_output["neg_labels"].value != -1).astype(np.float32)
        ),
    )


def get_negative_label_real(
    examples: GetSampleLabelOutput,
    target_label_emb: GetTargetLabelEmbeddingOutput,
    n_neg_labels: int,
    cls_distance: KGItemDistance,
    prop_distance: KGItemDistance,
):
    """
    Args:
        exlbls: example labels
        target_label_emb: target label embeddings
        n_neg_labels: number of negative labels to generate for training
        cls_distance: class distance
        prop_distance: property distance
    """
    id2index = target_label_emb["id2index"]
    disjoint_labels = get_sorted_labels(target_label_emb, cls_distance, prop_distance)

    # for each label, we retrieve the top K similar disjoint labels
    pos_label = []
    neg_labels = []

    for ctype in examples["column_type"].value:
        pos_ids = [id2index[c] for c in examples["original_column_type"].value[ctype]]
        neg_ids = [disjoint_labels[id][:n_neg_labels] for id in pos_ids]
        pos_label.append(pos_ids)
        # pos_label.append(",".join(str(x) for x in pos_ids))
        if len(neg_ids) == 1:
            neg_ids = neg_ids[0]
        else:
            tmp = neg_ids[0]
            # TODO: fix me, generate a better negative label -- perhaps automatically
            # generate negative labels -- use all neg_ids, for now, only the first one
            # is considered
            for i in range(tmp.shape[0]):
                if tmp[i] in pos_ids:
                    tmp[i] = -1
            # for i in range(1, len(neg_ids)):
            #     tmp = np.intersect1d(tmp, neg_ids[i])
            neg_ids = tmp
        neg_labels.append(neg_ids)

    max_n_pos_labels = max(len(x) for x in pos_label)
    for x in pos_label:
        if len(x) < max_n_pos_labels:
            x.extend([-1] * (max_n_pos_labels - len(x)))
    pos_label = np.asarray(pos_label)
    neg_labels = np.asarray(neg_labels)

    # pos_label = np.array([id2index[ctype] for ctype in exlbls["column_type"]])
    # neg_labels = disjoint_labels[pos_label][:, : args.n_neg_labels]

    return {
        "pos_labels": Single2DNumpyArray(pos_label),
        "neg_labels": Single2DNumpyArray(neg_labels),
    }


def get_sorted_labels(
    target_label_emb: GetTargetLabelEmbeddingOutput,
    cls_distance: KGItemDistance,
    prop_distance: KGItemDistance,
):
    labels = target_label_emb["labels"]
    id2index = target_label_emb["id2index"]
    classes_emb = target_label_emb["embeddings"]

    # for each class, we find the disjoint class and sort them by similarity provided by the embedding
    disjoint_labels = np.zeros((len(labels), len(labels) - 1), dtype=np.int32) - 1

    for mainlbl in labels:
        simlabels = []

        if isinstance(mainlbl, OntologyClass):
            fn_distance = cls_distance
        else:
            fn_distance = prop_distance

        for targetlbl in labels:
            if targetlbl.id == mainlbl.id:
                continue

            common_ancestors = set(mainlbl.ancestors.keys()).intersection(
                targetlbl.ancestors
            )

            if len(common_ancestors) == 0 or all(
                abs(dis) >= 2
                for dis in fn_distance.batch_get_distance(
                    [
                        (cls.id, anc)
                        for cls in [mainlbl, targetlbl]
                        for anc in common_ancestors
                    ]
                )
            ):
                if len(common_ancestors) > 0:
                    assert type(targetlbl) is type(mainlbl)
                simlabels.append(id2index[targetlbl.id])

        mainemb = classes_emb[id2index[mainlbl.id]]
        targetembs = classes_emb[simlabels]
        scores = np.dot(targetembs, mainemb) / (
            np.linalg.norm(mainemb) * np.linalg.norm(targetembs, axis=1)
        )

        simclasses_with_scores = sorted(
            zip(simlabels, scores), key=itemgetter(1), reverse=True
        )
        simlabels = [x[0] for x in simclasses_with_scores]

        disjoint_labels[id2index[mainlbl.id]][: len(simlabels)] = simlabels

    return disjoint_labels
