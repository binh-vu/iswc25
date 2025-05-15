from __future__ import annotations

import serde.csv
import serde.jl
from gpp.sem_label.feats import GetExamplesArgs, get_examples
from gpp.sem_label.feats._example import get_class_examples, get_prop_examples
from kgdata.wikidata.config import WikidataDirCfg
from kgdata.wikipedia.config import WikipediaDirCfg
from libactor.cache import IdentObj
from libactor.storage import GlobalStorage
from sm.namespaces.utils import KGName

from experiments.config import DATA_DIR, LIBACTOR_STORAGE_DIR
from experiments.dag import create_data_actor

GlobalStorage.init(LIBACTOR_STORAGE_DIR)
data_actor = create_data_actor()
kgdb = data_actor.db_actor.kgdbs[KGName.Wikidata]
ident_kgdb = IdentObj(kgdb.args.get_key(), kgdb)


WikidataDirCfg.init(DATA_DIR / "kgdata/wikidata/20240320")
WikipediaDirCfg.init(DATA_DIR / "kgdata/wikipedia/20230620")

missing_class_ids = [
    "Q108392111",
    "Q112099",
    "Q13107770",
    "Q13580678",
    "Q24243801",
    "Q3241045",
    "Q5119",
    "Q561068",
    "Q6644778",
    "Q67608976",
    "Q76514543",
    "Q828803",
    "Q85790993",
]
missing_prop_ids = [
    # "P11196",
    # "P1588",
    # "P1843",
    # "P2082",
    "P2341",
    # "P3068",
    "P37",
    "P58",
    # "P613",
    # "P6509",
    # "P6544",
    # "P6545",
    # "P6546",
    # "P9690",
]

entdb = kgdb.pydb.entity_metadata.cache()

# cls_lst_exs = get_class_examples(
#     entdb,
#     kgdb.kgname,
#     missing_class_ids,
#     args=GetExamplesArgs(
#         manual_example_dir=DATA_DIR / "manual-examples", only_manual_examples=False
#     ),
# )
# for clsid, exs in zip(missing_class_ids, cls_lst_exs):
#     outfile = DATA_DIR / "manual-examples" / f"classes/{clsid.lower()}.csv"
#     outfile.parent.mkdir(parents=True, exist_ok=True)
#     serde.csv.ser(
#         exs,
#         outfile,
#     )

prop_lst_exs = get_prop_examples(
    entdb,
    kgdb.ontology.value,
    missing_prop_ids,
    args=GetExamplesArgs(
        manual_example_dir=DATA_DIR / "manual-examples", only_manual_examples=False
    ),
)
for propid, exs in zip(missing_prop_ids, prop_lst_exs):
    # if kgdb.pydb.props[propid].is_object_property():
    outfile = DATA_DIR / "manual-examples" / f"props/{propid.lower()}.jl"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    serde.jl.ser(
        exs,
        outfile,
    )


# @Cache.flat_cache(
#         backend=Cache.file.jl(
#             mem_persist=True,
#             cls=EntityWithScore,
#             filename=lambda self, kgname, prop_id: f"{kgname}/object_props/{to_readable_filename(prop_id)}",
#         )
#     )
# @Cache.flat_cache(
#         backend=Cache.file.jl(
#             mem_persist=True,
#             cls=LiteralWithScore,
#             filename=lambda self, kgname, prop_id: f"{kgname}/data_props/{to_readable_filename(prop_id)}",
#         ),
#     )
# @Cache.flat_cache(
#         backend=Cache.file.csv(
#             mem_persist=True,
#             filename=lambda self, kgname, class_id: f"{kgname}/classes/{to_readable_filename(class_id)}",
#             deser_as_record=True,
#             dtype={"score": float},
#         )
#     )
