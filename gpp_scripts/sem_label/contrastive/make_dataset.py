from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import serde.json
import serde.pickle
import sm.outputs as O
from gp.actors.data import KGDB
from gpp.config import IDENT_PROPS
from gpp.llm.qa_llm import Schema
from gpp.sem_label.data_modules.v210 import SLabelV210DataModule
from gpp.sem_label.feats import get_text_embedding
from gpp.sem_label.feats._sample import AutoLabelSemanticModel
from gpp.sem_label.model_usage.contrastive_v220 import make_raw_dataset
from libactor.cache import IdentObj
from libactor.storage import GlobalStorage
from sm.dataset import Dataset
from sm_datasets import Datasets

from experiments.config import DATA_DIR, LIBACTOR_STORAGE_DIR
from experiments.dag import *


def make_v220_dataset(
    dataset_name_or_dir: str | Path, outfile: Path, fix_redirection: bool = True
):
    if GlobalStorage.instance is None:
        GlobalStorage.init(LIBACTOR_STORAGE_DIR)

    data_actor = create_data_actor()

    if isinstance(dataset_name_or_dir, str):
        dataset_name = dataset_name_or_dir
        examples = data_actor.forward(dataset_name).value
        kgdb = data_actor.get_kgdb(dataset_name)
    else:
        kgdb = data_actor.get_kgdb(dataset_name_or_dir.name)
        examples = load_dataset_from_disk(
            dataset_name_or_dir, kgdb, fix_redirection=fix_redirection
        )

    schema = Schema.from_ontology(kgdb.ontology.value, examples)

    raw_ds = make_raw_dataset(
        DATA_DIR / "datasets/ontology_examples",
        examples,
        IdentObj(kgdb.args.get_key(), kgdb),
        ignore_no_type_column=True,
        target_label_ids=sorted(schema.classes + schema.props),
        n_examples_per_label=150,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    serde.pickle.ser(raw_ds, outfile)
    return examples, raw_ds


def load_dataset_from_disk(
    dataset_name_or_dir: Path,
    kgdb: KGDB,
    fix_redirection: bool = True,
):
    examples = Dataset(dataset_name_or_dir).load()
    if fix_redirection:
        examples = Datasets().fix_redirection(
            examples,
            kgdb.pydb.entity_labels.cache(),
            kgdb.pydb.props.cache(),
            kgdb.pydb.entity_redirections.cache(),
            kgdb.kgns,
        )

    # if there is a auto-label file nearly, we need to load it
    # and add it to the examples
    if (dataset_name_or_dir / "autolabel.json").exists():
        # for ex in examples:
        #     ex =
        kgns = kgdb.kgns
        autolabel = serde.json.deser(dataset_name_or_dir / "autolabel.json")
        autolabel_edges = serde.json.deser(dataset_name_or_dir / "autolabel_edges.json")
        for ex in examples:
            sms = []
            for sm in ex.sms:
                newsm = AutoLabelSemanticModel.from_dict(sm.to_dict())
                if ex.id not in autolabel:
                    assert ex.id not in autolabel_edges
                    continue

                for col, coltypes in zip(
                    autolabel[ex.id]["entity_columns"],
                    autolabel[ex.id]["entity_column_types"],
                ):
                    ci = col[0]
                    uri2score = {}
                    for ctype in coltypes:
                        curi = kgns.id_to_uri(
                            kgdb.pydb.entity_redirections.get(
                                ctype["id"]["id"], ctype["id"]["id"]
                            )
                        )
                        uri2score[curi] = ctype["score"]

                    v = newsm.get_data_node(ci)
                    for inedge in newsm.in_edges(v.id):
                        u = newsm.get_node(inedge.source)
                        if inedge.abs_uri in IDENT_PROPS:
                            assert isinstance(u, O.ClassNode)
                            newsm.edge_probs[inedge.id] = uri2score[u.abs_uri]

                if ex.id in autolabel_edges and autolabel_edges[ex.id] is not None:
                    for prop in autolabel_edges[ex.id]:
                        if not newsm.has_data_node(
                            prop["source"]
                        ) or not newsm.has_data_node(prop["target"]):
                            continue
                        u = newsm.get_data_node(prop["source"])
                        v = newsm.get_data_node(prop["target"])

                        for inedge in newsm.in_edges(u.id):
                            if inedge.abs_uri in IDENT_PROPS:
                                pu = newsm.get_node(inedge.source)
                                for pred in prop["edges"]:
                                    puri = kgns.id_to_uri(
                                        kgdb.pydb.entity_redirections.get(
                                            pred["prop"], pred["prop"]
                                        )
                                    )
                                    if newsm.has_edge_between_nodes(pu.id, v.id, puri):
                                        newsm.edge_probs[
                                            newsm.get_edge_between_nodes(
                                                pu.id, v.id, puri
                                            ).id
                                        ] = (pred["freq"] * 2 - pred["unmatch_percent"])

                sms.append(newsm)
            ex.sms = sms
    return examples


def ensure_embedding(dataset, embedding_model):
    datamodule = SLabelV210DataModule(
        f"/tmp/{uuid4()}",  # we do not need data_dir in this case.
        embedding_model,
        get_text_embedding(embedding_model),
        train_batch_size=32,
        eval_batch_size=32,
        n_examples_per_column=100,
        n_examples_per_label=150,
    )

    dataset = datamodule.transformation("", dataset)
    dataset = datamodule.make_columnar_dataset("", dataset, embedding_readonly=False)
    datamodule.embedding_manager.flush(soft=False)


if __name__ == "__main__":
    from libactor.cache import IdentObj
    from libactor.storage import GlobalStorage

    from experiments.config import DATA_DIR, LIBACTOR_STORAGE_DIR
    from experiments.dag import create_data_actor

    GlobalStorage.init(LIBACTOR_STORAGE_DIR)
    data_actor = create_data_actor()
    kgdb = data_actor.db_actor.kgdbs[KGName.Wikidata]

    dataset = Path(
        "/Users/rook/workspace/projects/resm-v2/data/datasets/wiki-20230620/auto-label/wt-limited-easy-sp51/wt-limited-easy-sp51-v1"
    )
    # exs = load_dataset_from_disk(dataset, kgdb)
    ds = make_v220_dataset(
        dataset,
        DATA_DIR / f"experiments/traindata_v220/{dataset.name}.pkl",
        fix_redirection=False,
    )
    ensure_embedding(ds, "BAAI/bge-m3")
