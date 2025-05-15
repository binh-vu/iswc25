import os

os.environ["CACHE_GPP_SEM_LABEL"] = "0"
os.environ["CACHE_GPP_SEM_MODEL"] = "0"

from typing import Annotated

import serde.json
import typer
from gpp.sem_label.model_usage.contrastive_v220 import *
from libactor.storage import GlobalStorage
from timer import Timer

from experiments.config import DATA_DIR, DATABASE_DIR, LIBACTOR_STORAGE_DIR, ROOT_DIR
from experiments.dag import *

GlobalStorage.init(LIBACTOR_STORAGE_DIR)

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_enable=False)


@app.command()
def main(
    precompute_dir: Annotated[
        str,
        typer.Option(
            "--precompute-sem-label", help="Path to the precompute sem-label dir"
        ),
    ],
):
    data_actor = create_data_actor()

    dataset_name = "wt250"
    kgdb = data_actor.get_kgdb(dataset_name)
    examples = data_actor.forward(dataset_name)

    from gp.actors.el.canreg import CanRegActor, CanRegActorArgs
    from gp.entity_linking.candidate_recognition import HeuristicCanReg
    from libactor.dag import Flow
    from sm.misc.funcs import get_classpath

    dag = create_gpp_dag(
        dictmap={
            # "table": [get_table, remove_unknown_columns],
            "table": get_table,
            "canreg": Flow(
                "table",
                CanRegActor(
                    CanRegActorArgs(
                        clspath=get_classpath(HeuristicCanReg),
                        clsargs={},
                    )
                ),
            ),
            "semlabel": Flow(
                "table",
                GppSemLabelActor(
                    GppSemLabelArgs(
                        model="gpp.sem_label.models.precompute_model.PrecomputeModel",
                        model_args={
                            "predict_files": [
                                DATA_DIR
                                / f"experiments/sem-label-models/{precompute_dir}/wt250.json",
                                DATA_DIR
                                / f"experiments/sem-label-models/{precompute_dir}/t2dv2.json",
                            ]
                        },
                        data="gpp.sem_label.models.precompute_model.get_dataset",
                        data_args={},
                    )
                ),
            ),
            "graphspace": GraphSpaceV1Actor(
                GraphSpaceV1Args(
                    top_k_data_props=5,
                    top_k_object_props=5,
                )
            ),
            "sm": Flow(
                ["table", "semlabel", "canreg", "graphspace"],
                GppSemModelActor(
                    GppSemModelArgs(
                        algo="gpp.sem_model.from_sem_label.algo_v300.AlgoV301",
                        algo_args={"coeff_stscore": 3, "coeff_kgscore": 2},
                    )
                ),
            ),
        }
    )

    contextfn = get_gpp_context(data_actor, dataset_name)
    graphspace = data_actor.forward(dataset_name)

    with Timer().watch_and_report(f"Predict {dataset_name}"):
        output = dag.par_process(
            [{"table": (ex,), "graphspace": (graphspace,)} for ex in examples.value],
            {"sm", "table"},
            [contextfn for _ in examples.value],
            n_jobs=-1,
        )

    # validate if the order is correct
    assert [ex.id for ex in examples.value] == [
        exout["table"][0].id for exout in output
    ]

    outfile = (
        DATA_DIR
        / f"experiments/e01_gpp_main/gpp/precomputed/{precompute_dir}/{dataset_name}.json"
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    serde.json.ser(
        {x["table"][0].id: x["sm"][0].value.to_dict() for x in output},
        outfile,
        indent=2,
    )

    evaluator = Evaluator(kgdb.ontology.value, kgdb.pydb.entity_labels.cache())

    ctas = evaluator.avg_cta(examples.value, [x["sm"][0].value for x in output])
    cpas = evaluator.avg_cpa(examples.value, [x["sm"][0].value for x in output])


if __name__ == "__main__":
    app()
