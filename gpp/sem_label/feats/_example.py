from __future__ import annotations

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, cast
from urllib.parse import urlparse

import orjson
import serde.csv
import serde.jl
from kgdata.db import EntityMetadata
from kgdata.dbpedia.datasets.entities import entities as dbpedia_entities
from kgdata.dbpedia.datasets.entity_types_and_degrees import (
    entity_types_and_degrees as dbpedia_entity_types_and_degrees,
)
from kgdata.dbpedia.datasets.properties import properties as dbpedia_properties
from kgdata.misc.ntriples_parser import node_from_dict, node_to_dict
from kgdata.models import MultiLingualString, MultiLingualStringList, Ontology
from kgdata.models.entity import Entity
from kgdata.models.multilingual import MultiLingualString, MultiLingualStringList
from kgdata.spark.common import get_spark_context
from kgdata.wikidata.datasets.entities import entities as wikidata_entities
from kgdata.wikidata.datasets.entity_types_and_degrees import EntityTypeAndDegree
from kgdata.wikidata.datasets.entity_types_and_degrees import (
    entity_types_and_degrees as wikidata_entity_types_and_degrees,
)
from kgdata.wikidata.datasets.properties import properties as wikidata_properties
from kgdata.wikidata.models import WDEntity, WDEntityMetadata, WDValueKind
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdvalue import WDValue, WDValueKind
from rdflib import Literal, URIRef
from sm.inputs.link import EntityId, EntityIdWithScore
from sm.misc.funcs import filter_duplication
from sm.namespaces.utils import KGName

EntityDB = Mapping[str, EntityMetadata | WDEntityMetadata]
LiteralValue = WDValueKind | Literal


@dataclass
class GetExamplesArgs:
    k: int = field(default=1000, metadata={"help": "The number of examples to return."})
    max_distance: int = field(
        default=3,
        metadata={
            "help": "The maximum distance of an item type to be considered instance of the class"
        },
    )
    respect_range_constraints: bool = field(
        default=True,
        metadata={
            "help": "If True, the values of property will comply with the property's ranges"
        },
    )
    manual_example_dir: Optional[Path] = field(
        default=None,
        metadata={
            "help": "The directory that contains manually created examples -- this has higher priority"
        },
    )
    no_empty_examples: bool = field(
        default=True,
        metadata={
            "help": "If True, raise an exception if the class/property has no examples"
        },
    )
    only_manual_examples: bool = field(
        default=True,
        metadata={
            "help": "If True, only return examples from the manual example directory"
        },
    )


@dataclass
class LiteralWithScore:
    source: str
    value: LiteralValue
    score: float

    def to_dict(self):
        return {
            "source": self.source,
            "value": (
                node_to_dict(self.value)
                if isinstance(self.value, Literal)
                else self.value.to_dict()
            ),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, obj: dict):
        if obj["value"]["type"] == "Literal":
            value = node_from_dict(obj["value"])
            assert isinstance(value, Literal)
        else:
            value = WDValue(**obj["value"])

        return cls(source=obj["source"], value=value, score=obj["score"])


@dataclass
class EntityWithScore:
    id: str
    label: MultiLingualString
    description: MultiLingualString
    aliases: MultiLingualStringList
    source: str  # id of the source entity containing this value
    score: float

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label.to_dict(),
            "description": self.description.to_dict(),
            "aliases": self.aliases.to_dict(),
            "source": self.source,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            id=obj["id"],
            label=MultiLingualString.from_dict(obj["label"]),
            description=MultiLingualString.from_dict(obj["description"]),
            aliases=MultiLingualStringList.from_dict(obj["aliases"]),
            source=obj["source"],
            score=obj["score"],
        )


GetExamplesOutput = Sequence[Sequence[EntityWithScore] | Sequence[LiteralWithScore]]


def get_examples(
    ids: list[str],
    entdb: EntityDB,
    ontology: Ontology,
    args: GetExamplesArgs,
) -> GetExamplesOutput:
    """Get examples of an ontology class or property."""
    class_id_index = []
    class_ids = []
    prop_id_index = []
    prop_ids = []

    output: list[Sequence[EntityWithScore] | Sequence[LiteralWithScore]] = [
        [] for _ in ids
    ]

    if args.manual_example_dir is not None:
        kgexdir = args.manual_example_dir / str(ontology.kgname)
    else:
        kgexdir = None

    for i, id in enumerate(ids):
        is_id_cls = id in ontology.classes
        if args.manual_example_dir is not None:
            assert kgexdir is not None
            files = list(
                kgexdir.glob(
                    ("classes-*" if is_id_cls else "props-*")
                    + f"/{to_readable_filename(id).lower()}.*"
                )
            )
            if len(files) > 0:
                file = files[0]
                if id in ontology.classes:
                    output[i] = parse_class_examples(
                        entdb,
                        serde.csv.deser(
                            file, deser_as_record=True, dtype={"score": float}
                        ),
                    )
                elif ontology.props[id].is_object_property():
                    output[i] = serde.jl.deser(
                        file,
                        cls=EntityWithScore,
                    )
                else:
                    output[i] = serde.jl.deser(
                        file,
                        cls=LiteralWithScore,
                    )
                continue

            if args.only_manual_examples and len(output[i]) == 0:
                raise Exception(f"Manual examples for {id} not found")

        if is_id_cls:
            class_id_index.append(i)
            class_ids.append(id)
        else:
            prop_id_index.append(i)
            prop_ids.append(id)

    if len(class_ids) > 0:
        unk_class_id_index = [idx for idx in class_id_index if len(output[idx]) == 0]
        unk_class_ids = [class_ids[idx] for idx in unk_class_id_index]

        for idx, values in zip(
            unk_class_id_index,
            get_class_examples(entdb, ontology.kgname, unk_class_ids, args),
        ):
            if args.no_empty_examples:
                assert len(values) > 0, f"Class {ids[idx]} has no examples"
            output[idx] = parse_class_examples(entdb, values)
    if len(prop_ids) > 0:
        unk_prop_id_index = [idx for idx in prop_id_index if len(output[idx]) == 0]
        unk_prop_ids = [prop_ids[idx] for idx in unk_prop_id_index]

        for idx, values in zip(
            unk_prop_id_index, get_prop_examples(entdb, ontology, unk_prop_ids, args)
        ):
            if args.no_empty_examples:
                assert len(values) > 0, f"Property {ids[idx]} has no examples"
            output[idx] = values
    return output


def get_class_examples(
    entdb: EntityDB,
    kgname: KGName,
    class_ids: list[str],
    args: GetExamplesArgs,
):
    out: list[list[dict]] = []

    for cls_examples in get_class_most_popular_entity_ids(
        kgname,
        class_ids,
        max_distance=args.max_distance,
        far_distance=args.max_distance + 100,
        max_num_items=args.k,
    ):
        out_ex = []
        for cls_ex in cls_examples:
            ent = entdb[str(cls_ex.id)]
            out_ex.append(
                {
                    "id": str(cls_ex.id),
                    "label": str(ent.label),
                    "source": str(cls_ex.source),
                    "score": cls_ex.score,
                }
            )
        out.append(out_ex)
    return out


def get_prop_examples(
    entdb: EntityDB, ontology: Ontology, prop_ids: list[str], args: GetExamplesArgs
):
    propdb = ontology.props
    object_prop_ids = [pid for pid in prop_ids if propdb[pid].is_object_property()]
    data_prop_ids = [pid for pid in prop_ids if propdb[pid].is_data_property()]

    object_prop_examples = dict(
        zip(
            object_prop_ids,
            get_object_prop_examples(entdb, ontology, object_prop_ids, args),
        )
    )
    data_prop_examples = dict(
        zip(data_prop_ids, get_data_prop_examples(ontology, data_prop_ids, args))
    )

    return [
        (
            object_prop_examples.get(pid, [])
            if pid in object_prop_examples
            else data_prop_examples.get(pid, [])
        )
        for pid in prop_ids
    ]


def get_object_prop_examples(
    entdb: EntityDB, ontology: Ontology, prop_ids: list[str], args: GetExamplesArgs
):
    assert all(ontology.props[pid].is_object_property() for pid in prop_ids)
    out: list[list[EntityWithScore]] = []

    for prop_examples in get_object_prop_most_popular_values(
        ontology.kgname,
        prop_ids,
        max_distance=args.max_distance,
        far_distance=args.max_distance + 100,
        max_num_items=args.k,
        respect_range_constraints=args.respect_range_constraints,
    ):
        out_ex = []
        for prop_ex in prop_examples:
            ent = entdb[str(prop_ex.id)]
            out_ex.append(
                EntityWithScore(
                    id=str(prop_ex.id),
                    label=ent.label,
                    description=ent.description,
                    aliases=ent.aliases,
                    source=prop_ex.source,
                    score=prop_ex.score,
                )
            )
        out.append(out_ex)
    return out


def get_data_prop_examples(
    ontology: Ontology, prop_ids: list[str], args: GetExamplesArgs
):
    assert all(ontology.props[pid].is_data_property() for pid in prop_ids)
    return get_data_prop_most_popular_values(
        ontology.kgname, prop_ids, max_num_items=args.k
    )


def parse_class_examples(entdb: EntityDB, examples: list[dict]):
    out: list[EntityWithScore] = []
    for ex in examples:
        ent = entdb[ex["id"]]
        out.append(
            EntityWithScore(
                id=ent.id,
                label=ent.label,
                description=ent.description,
                aliases=ent.aliases,
                source=ex["source"],
                score=ex["score"],
            )
        )
    return out


def get_class_most_popular_entity_ids(
    kgname: KGName,
    class_ids: list[str],
    max_distance: int = 3,
    far_distance: int = 100,
    max_num_items: int = 1000,
    scoring_with_type: bool = True,
) -> list[list[ValueWithScore]]:
    if kgname == KGName.Wikidata:
        entity_types_and_degrees_dataset = wikidata_entity_types_and_degrees()
    elif kgname == KGName.DBpedia:
        entity_types_and_degrees_dataset = dbpedia_entity_types_and_degrees()
    else:
        raise NotImplementedError(kgname)

    set_class_ids = set(class_ids)

    def filter_in_class(ent: EntityTypeAndDegree):
        for cid in set_class_ids:
            if ent.types.get(cid, far_distance) <= max_distance:
                return True
        return False

    def unpack_class(ent: EntityTypeAndDegree):
        output: list[tuple[str, ValueWithScore]] = []
        for cid in set_class_ids:
            dis = ent.types.get(cid, far_distance)
            if dis > max_distance:
                continue

            output.append(
                (
                    cid,
                    ValueWithScore(
                        id=ent.id,
                        source=ent.id,
                        score=(
                            EntityPopularity.get_ent_score_of_type(ent, cid)
                            if scoring_with_type
                            else EntityPopularity.get_ent_score(ent)
                        ),
                    ),
                )
            )
        return output

    out = dict(
        entity_types_and_degrees_dataset.get_rdd()
        .filter(filter_in_class)
        .flatMap(unpack_class)
        .combineByKey(
            TopKValueAggregator.create_combiner,
            partial(TopKValueAggregator.merge_value, max_num_items),
            partial(TopKValueAggregator.merge_combiner, max_num_items),
        )
        .collect()
    )
    return [out.get(cid, []) for cid in class_ids]


def get_object_prop_most_popular_values(
    kgname: KGName,
    prop_ids: list[str],
    max_distance: int = 3,
    far_distance: int = 100,
    max_num_items: int = 1000,
    respect_range_constraints: bool = True,
) -> list[list[ValueWithScore]]:
    """Get values of object prop and sort them by their popularity (combination of in/out degree)"""
    set_prop_ids = set(prop_ids)

    if len(set_prop_ids) == 0:
        return []

    def extract_wd_value(ent: WDEntity):
        out: list[tuple[str, tuple[str, str]]] = []
        for pid in set_prop_ids:
            for s in ent.props.get(pid, []):
                if s.value.is_entity_id(s.value):
                    out.append((s.value.as_entity_id_safe(), (pid, ent.id)))
        return filter_duplication(out)

    def extract_dbp_value(ent: Entity):
        out: list[tuple[str, tuple[str, str]]] = []
        for pid in set_prop_ids:
            for s in ent.props.get(pid, []):
                if isinstance(s.value, URIRef):
                    out.append((str(s.value), (pid, ent.id)))
        return filter_duplication(out)

    pid2ranges = None

    if kgname == KGName.Wikidata:
        entity_types_and_degrees_dataset = wikidata_entity_types_and_degrees()
        dataset = wikidata_entities().get_rdd().flatMap(extract_wd_value)
        if respect_range_constraints:
            pid2ranges = get_spark_context().broadcast(
                {p.id: p.ranges for p in wikidata_properties().get_list()}
            )
    elif kgname == KGName.DBpedia:
        entity_types_and_degrees_dataset = dbpedia_entity_types_and_degrees()
        dataset = dbpedia_entities().get_rdd().flatMap(extract_dbp_value)
        if respect_range_constraints:
            pid2ranges = get_spark_context().broadcast(
                {p.id: p.ranges for p in dbpedia_properties().get_list()}
            )
    else:
        raise NotImplementedError(kgname)

    def combine_join_res(
        tup: tuple[str, tuple[Iterable[tuple[str, str]], tuple[dict[str, int], float]]],
    ):
        entid, (pidwsources, (enttypes, score)) = tup
        entwscore = EntityIdWithScore(EntityId(entid, str(kgname)), score)
        if not respect_range_constraints:
            return [
                (pid, ValueWithScore(id=entid, source=source_ent, score=score))
                for pid, source_ent in pidwsources
            ]

        assert pid2ranges is not None
        output = []
        for pid, source_ent in pidwsources:
            ranges = pid2ranges.value[pid]
            if len(ranges) == 0 or any(
                enttypes.get(range, far_distance) <= max_distance for range in ranges
            ):
                output.append(
                    (pid, ValueWithScore(id=entid, source=source_ent, score=score))
                )
        return output

    out = dict(
        dataset.groupByKey()
        .join(
            entity_types_and_degrees_dataset.get_rdd().map(
                lambda e: (e.id, (e.types, EntityPopularity.get_ent_score(e)))
            )
        )
        .flatMap(combine_join_res)
        .combineByKey(
            TopKValueAggregator.create_combiner,
            partial(TopKValueAggregator.merge_value_unique, max_num_items),
            partial(TopKValueAggregator.merge_combiner_unique, max_num_items),
        )
        .collect()
    )
    return [out.get(pid, []) for pid in prop_ids]


def get_data_prop_most_popular_values(
    kgname: KGName,
    prop_ids: list[str],
    max_num_items: int = 1000,
) -> list[list[LiteralWithScore]]:
    """Get values of data prop and sort them by the popularity of the entities that have the value. The popularity
    of an entity is a combination of its in/out degree.
    """

    def extract_wd_value(ent: WDEntity):
        out: dict[str, set[str]] = defaultdict(set)
        for pid in set_prop_ids:
            for s in ent.props.get(pid, []):
                if s.value.is_entity_id(s.value):
                    continue
                out[pid].add(orjson.dumps(s.value.to_dict()).decode())
        return (ent.id, dict(out))

    def extract_dbp_value(ent: Entity):
        out: dict[str, set[str]] = defaultdict(set)
        for pid in set_prop_ids:
            for s in ent.props.get(pid, []):
                if isinstance(s.value, Literal):
                    out[pid].add(orjson.dumps(node_to_dict(s.value)).decode())
        return (ent.id, dict(out))

    def combine_join_res(tup: tuple[str, tuple[dict[str, set[str]], float]]):
        entid, (pid2values, entscore) = tup
        out: list[tuple[str, ValueWithScore]] = []
        for pid, values in pid2values.items():
            for value in values:
                out.append(
                    (pid, ValueWithScore(id=value, source=entid, score=entscore))
                )
        return out

    if kgname == KGName.Wikidata:
        entity_types_and_degrees_dataset = wikidata_entity_types_and_degrees()
        dataset = wikidata_entities().get_rdd().map(extract_wd_value)
        deser_val = lambda x: WDValue(**orjson.loads(x))
    elif kgname == KGName.DBpedia:
        entity_types_and_degrees_dataset = dbpedia_entity_types_and_degrees()
        dataset = dbpedia_entities().get_rdd().map(extract_dbp_value)
        deser_val = lambda x: cast(Literal, node_from_dict(orjson.loads(x)))
    else:
        raise NotImplementedError(kgname)
    set_prop_ids = set(prop_ids)

    out = dict(
        dataset.join(
            entity_types_and_degrees_dataset.get_rdd().map(
                lambda e: (e.id, EntityPopularity.get_ent_score(e))
            )
        )
        .flatMap(combine_join_res)
        .combineByKey(
            TopKValueAggregator.create_combiner,
            partial(TopKValueAggregator.merge_value_unique, max_num_items),
            partial(TopKValueAggregator.merge_combiner_unique, max_num_items),
        )
        .collect()
    )

    return [[val.to_literal(deser_val) for val in out.get(pid, [])] for pid in prop_ids]


@dataclass
class ValueWithScore:
    id: str
    source: str  # id of the source entity containing this value
    score: float

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            id=obj["id"],
            source=obj["source"],
            score=obj["score"],
        )

    def to_literal(self, deser_val: Callable[[str], LiteralValue]):
        return LiteralWithScore(
            source=self.source, value=deser_val(self.id), score=self.score
        )


def to_readable_filename(class_id: str):
    if class_id.startswith("http"):
        parsedurl = urlparse(class_id)
        if parsedurl.netloc == "dbpedia.org":
            if parsedurl.path.startswith("/ontology/"):
                class_id = re.sub(r"[^a-zA-Z0-9]", "_", parsedurl.path[10:])
            else:
                class_id = re.sub(r"[^a-zA-Z0-9]", "_", parsedurl.path)
            class_id = re.sub(r"_+", "_", class_id)
            return class_id
        if parsedurl.netloc == "minmod.isi.edu":
            if parsedurl.path.startswith("/ontology-simple/"):
                class_id = re.sub(r"[^a-zA-Z]", "_", parsedurl.path[17:])
                return class_id
        raise NotImplementedError()

    return class_id


class EntityPopularity:
    @staticmethod
    def get_ent_score_of_type(ent: EntityTypeAndDegree, class_id: str):
        outscale, inscale = 0.95, 0.05
        degree_scale = 0.1

        wp_score = (ent.wikipedia_outdegree or 0.0) * outscale + (
            ent.wikipedia_indegree or 0.0
        ) * inscale
        db_score = ent.outdegree * outscale + ent.indegree * inscale

        dist = ent.types[class_id]
        if dist <= 1:
            dist_score = -dist * 500
        elif dist <= 2:
            dist_score = -dist * 5000
        elif dist <= 3:
            dist_score = -dist * 10000
        else:
            dist_score = -dist * 20000
        return (wp_score + db_score) * degree_scale + dist_score

    @staticmethod
    def get_ent_score(ent: EntityTypeAndDegree):
        outscale, inscale = 0.95, 0.05
        wp_score = (ent.wikipedia_outdegree or 0.0) * outscale + (
            ent.wikipedia_indegree or 0.0
        ) * inscale
        db_score = ent.outdegree * outscale + ent.indegree * inscale
        return wp_score + db_score


class TopKValueAggregator:
    @staticmethod
    def create_combiner(val: ValueWithScore):
        return [val]

    @staticmethod
    def merge_value_unique(
        max_num_items: int, lst: list[ValueWithScore], val: ValueWithScore
    ):
        for i, x in enumerate(lst):
            if x.id == val.id:
                # handle duplicated values -- they have duplicated value doesn't mean they have the same score
                # it can be from different entities
                if x.score < val.score:
                    lst[i] = val
                    lst.sort(key=lambda v: v.score, reverse=True)
                return lst

        bisect.insort(lst, val, key=lambda x: -x.score)
        if len(lst) > max_num_items:
            lst = lst[:max_num_items]
        return lst

    @staticmethod
    def merge_value(max_num_items: int, lst: list[ValueWithScore], val: ValueWithScore):
        bisect.insort(lst, val, key=lambda x: -x.score)
        if len(lst) > max_num_items:
            lst = lst[:max_num_items]
        return lst

    @staticmethod
    def merge_combiner(
        max_num_items: int, lst1: list[ValueWithScore], lst2: list[ValueWithScore]
    ):
        lst1.extend(lst2)
        lst1 = sorted(lst1, key=lambda x: -x.score)[:max_num_items]
        return lst1

    @staticmethod
    def merge_combiner_unique(
        max_num_items: int, lst1: list[ValueWithScore], lst2: list[ValueWithScore]
    ):
        exist_vals = {x.id: i for i, x in enumerate(lst1)}
        has_better_dup = False
        for val in lst2:
            if val.id not in exist_vals:
                bisect.insort(lst1, val, key=lambda x: -x.score)
            else:
                # handle duplicated value
                i = exist_vals[val.id]
                if lst1[i].score < val.score:
                    lst1[i] = val
                    has_better_dup = True

        if has_better_dup:
            lst1.sort(key=lambda x: x.score, reverse=True)

        if len(lst1) > max_num_items:
            lst1 = lst1[:max_num_items]

        return lst1
