from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from gpp.sem_label.feats._example import (
    EntityWithScore,
    GetExamplesOutput,
    LiteralWithScore,
)
from gpp.sem_label.feats._negative_label import GetNegativeLabelOutput
from gpp.sem_label.feats._target_example import format_globe_coordinate
from gpp.sem_label.feats._target_label_embedding import GetTargetLabelEmbeddingOutput
from keyvec import BatchText, EmbeddingManager
from kgdata.models import Ontology
from kgdata.wikidata.models.wdvalue import WDValue
from rdflib import Literal
from smml.dataset import Feat


class GetTargetLabelExampleEmbeddingOutput(TypedDict):
    pos_out: GetTargetLabelExampleFnOutput
    pos_labels: GetTargetLabelExampleFnOutputExEmbeddings
    pos_labels_mask: GetTargetLabelExampleFnOutputExMask
    neg_out: GetTargetLabelExampleFnOutput
    neg_labels: GetTargetLabelExampleFnOutputExEmbeddings
    neg_labels_mask: GetTargetLabelExampleFnOutputExMask


def get_target_label_example_embedding(
    target_labels_emb: GetTargetLabelEmbeddingOutput,
    target_label_examples: GetExamplesOutput,
    neglabel: GetNegativeLabelOutput,
    ontology: Ontology,
    text_emb: EmbeddingManager,
    n_examples_per_label: int,
):
    """Generate embedding for examples of target labels.

    Args:
        n_examples_per_label: The number of examples per label
    """
    kgname = ontology.kgname

    target_labels = target_labels_emb
    id2label = {l.id: l for l in target_labels["labels"]}

    # get embeddings of example values of each target label
    batch_text = BatchText()

    # convert the embeddings of representative into pos/neg examples
    label2examples = np.zeros(
        (len(target_labels["labels"]), n_examples_per_label), dtype=np.int32
    )
    mask_label2examples = np.ones(
        (len(target_labels["labels"]), n_examples_per_label), dtype=np.int32
    )

    for label_idx, label, lbl_examples in zip(
        range(len(target_labels["labels"])),
        target_labels["labels"],
        target_label_examples,
    ):
        assert len(lbl_examples) > 0, f"Label {label.id} has no examples"

        # convert examples into texts to add to embeddings
        encoded_examples = []
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

            encoded_examples.append(batch_text.add_text(text))
        if len(encoded_examples) < n_examples_per_label:
            mask_label2examples[label_idx][len(encoded_examples) :] = 0
            encoded_examples.extend(
                [0] * (n_examples_per_label - len(encoded_examples))
            )

        label2examples[label_idx] = encoded_examples

    label_example_embeddings = text_emb.batch_get(batch_text)
    assert not np.any(np.isnan(label_example_embeddings))

    neglabel_pos_labels = neglabel["pos_labels"].value
    # neglabel_pos_labels_mask = neglabel["pos_labels_mask"].value
    neglabel_neg_labels = neglabel["neg_labels"].value
    # neglabel_neg_labels_mask = neglabel["neg_labels_mask"].value

    assert (
        isinstance(neglabel_pos_labels, np.ndarray)
        and isinstance(neglabel_neg_labels, np.ndarray)
        # and isinstance(neglabel_pos_labels_mask, np.ndarray)
        # and isinstance(neglabel_neg_labels_mask, np.ndarray)
    )

    pos_output = GetTargetLabelExampleFnOutput(
        id2index=batch_text.unique_text,
        embeddings=label_example_embeddings,
        label_examples=label2examples,
        mask_label_examples=mask_label2examples,
        sample_labels=neglabel_pos_labels,
        # sample_labels_mask=neglabel_pos_labels_mask,
    )
    neg_output = GetTargetLabelExampleFnOutput(
        id2index=batch_text.unique_text,
        embeddings=label_example_embeddings,
        label_examples=label2examples,
        mask_label_examples=mask_label2examples,
        sample_labels=neglabel_neg_labels,
        # sample_labels_mask=neglabel_neg_labels_mask,
    )

    return GetTargetLabelExampleEmbeddingOutput(
        pos_out=pos_output,
        pos_labels=pos_output.get_ex_embeddings(),
        pos_labels_mask=pos_output.get_ex_mask(),
        neg_out=neg_output,
        neg_labels=neg_output.get_ex_embeddings(),
        neg_labels_mask=neg_output.get_ex_mask(),
    )


@dataclass
class GetTargetLabelExampleFnOutput(Feat):
    id2index: dict[str, int]
    embeddings: np.ndarray  # examples x emb_dim
    label_examples: np.ndarray  # n lables x n examples (dtype int)
    mask_label_examples: np.ndarray  # n labels x n examples (dtype float)
    sample_labels: np.ndarray  # n training/testing samples x n classes (-1 if missing)
    # sample_labels_mask: np.ndarray  # n training/testing examples x n classes (dtype bool)

    # def select_labels(self, labels: list[str] | list[int]):
    #     assert len(labels) > 0
    #     if isinstance(labels[0], int):
    #         label_indices = cast(list[int], labels)
    #         index2id = {v: k for k, v in self.id2index.items()}
    #         label_ids = [index2id[l] for l in label_indices]
    #     else:
    #         label_ids: list[str] = cast(list[str], labels)
    #         label_indices = [self.id2index[l] for l in label_ids]

    #     # TODO: it's a bit complicated to get the embeddings correctly
    #     raise NotImplementedError()
    #     # return GetTargetLabelExampleFnOutput(
    #     #     id2index={k: i for i, k in enumerate(label_ids)},
    #     #     embeddings=
    #     # )

    def __getitem__(self, idx: int | slice):
        raise Exception("Should not use this feature directly.")

    def __len__(self):
        return len(self.sample_labels)

    def get_ex_embeddings(self):
        return GetTargetLabelExampleFnOutputExEmbeddings(
            embeddings=self.embeddings,
            label_examples=self.label_examples,
            example_labels=self.sample_labels,
        )

    def get_ex_mask(self):
        return GetTargetLabelExampleFnOutputExMask(
            mask_label_examples=self.mask_label_examples,
            sample_labels=self.sample_labels,
            # sample_labels_mask=self.sample_labels_mask,
        )


@dataclass
class GetTargetLabelExampleFnOutputExEmbeddings(Feat):
    embeddings: np.ndarray  # examples x emb_dim
    label_examples: np.ndarray  # n labels x n examples (dtype int)
    example_labels: (
        np.ndarray
    )  # n training/testing examples x n classes (-1 if missing)
    # we do not the sample_labels_mask because it will be take care by the GetTargetLabelExampleFnOutputExMask

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return GetTargetLabelExampleFnOutputExEmbeddings(
                embeddings=self.embeddings,
                label_examples=self.label_examples,
                example_labels=self.example_labels[idx],
            )

        # n classes x n examples
        selected_label_examples = self.label_examples[self.example_labels[idx]]
        # n classes x n examples x emb_dim
        return self.embeddings[selected_label_examples]

    def __len__(self):
        return len(self.example_labels)


@dataclass
class GetTargetLabelExampleFnOutputExMask(Feat):
    mask_label_examples: np.ndarray  # n labels x n examples (dtype float)
    sample_labels: np.ndarray  # n training/testing samples x n classes (-1 if missing)
    # sample_labels_mask: np.ndarray  # n training/testing examples x n classes (dtype bool)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return GetTargetLabelExampleFnOutputExMask(
                mask_label_examples=self.mask_label_examples,
                sample_labels=self.sample_labels[idx],
                # sample_labels_mask=self.sample_labels_mask[idx],
            )

        # n classes x n samples
        # return self.mask_label_examples[self.example_labels[idx]] * np.expand_dims(
        #     self.sample_labels_mask[idx], axis=1
        # )

        # we do not need to multiply with the mask -- the reason is that it will make it difficult to calculate
        # mean over the number of examples (will got nan) -- this will be taken care of by the label mask.
        return self.mask_label_examples[self.sample_labels[idx]]

    def __len__(self):
        return len(self.sample_labels)
