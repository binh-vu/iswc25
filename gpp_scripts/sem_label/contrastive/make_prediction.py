from __future__ import annotations

import shutil
from ast import Global
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import serde.json
import typer
from experiments.config import DATA_DIR, LIBACTOR_STORAGE_DIR
from libactor.storage import GlobalStorage
from loguru import logger
from timer import Timer

from gpp.sem_label.data_modules.base import get_best_model
from gpp.sem_label.feats import get_text_embedding

"""
This script is used to make predictions using a trained model. The predictions are saved to a file so that we can load and evaluate them later.
"""

pl_rootdir = DATA_DIR / "lightning/csv/sem-label"

app = typer.Typer(name="Make Prediction")


@app.command()
def predict(
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="Name of our model"),
    ],
    ckpt_file_pattern: Annotated[
        str,
        typer.Option("--ckpt-file", help="relative path (glob) to the checkpoint file"),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Path to the dataset directory"),
    ],
    datasets: Annotated[
        list[str], typer.Option("--datasets", help="List of datasets to predict")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Path to the output directory")
    ],
):
    """
    Make predictions using the specified model and version, and save them to a file.

    Checkout how to use this function in `experiments/setups/e01_gpp_main/prediction.sh`.
    """
    timer = Timer()
    if model_name.startswith("contrastive-v220"):
        from gpp.sem_label.model_usage.contrastive_v220 import (
            SemLabelV220,
            SLabelV210DataModule,
        )

        (ckpt_file,) = list(pl_rootdir.glob(ckpt_file_pattern))

        datamodule_params = serde.json.deser(
            ckpt_file.parent.parent / "datamodule_params.json"
        )

        datamodule = SLabelV210DataModule(
            dataset_dir,
            datamodule_params["model_name_or_path"],
            get_text_embedding(
                datamodule_params["embedding_manager"]["embedding_model"],
                datamodule_params["embedding_manager"]["customization"],
            ),
            train_batch_size=32,
            eval_batch_size=32,
        )

        if model_name == "contrastive-v220":
            clspath = "gpp.sem_label.models.contrastive.v220.V220"
        elif model_name == "contrastive-v220a":
            clspath = "gpp_scripts.sem_label.contrastive.train.ablation_study.v221a_without_column_header.V221a"
        elif model_name == "contrastive-v220b":
            clspath = "gpp_scripts.sem_label.contrastive.train.ablation_study.v221b_without_column_value.V221b"
        elif model_name == "contrastive-v220c":
            clspath = "gpp_scripts.sem_label.contrastive.train.ablation_study.v221c_without_multi_task.V221c"
        else:
            raise NotImplementedError(model_name)

        model = SemLabelV220.load(Path(f"/tmp/{uuid4()}"), ckpt_file, clspath)
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name in datasets:
            logger.info(f"Predicting {dataset_name}")
            with timer.watch_and_report(f"Predicting {dataset_name}"):
                dataset = datamodule.load_dataset(dataset_name)
                output = model.predict_dataset(dataset, verbose=True, evaluate=True)
                serde.json.ser(output, output_dir / (dataset_name + ".json"), indent=2)
    else:
        raise NotImplementedError(model_name)


@app.command()
def export_sem_label_model(
    version: Annotated[int, typer.Option("--version", help="Version of our model")],
    metric: Annotated[
        str, typer.Option("--metric", help="Metric to use to select the best model")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Path to the output directory")
    ],
    smaller_is_better: Annotated[
        bool,
        typer.Option(
            "--smaller-is-better",
            help="Whether smaller values of the metric are better",
        ),
    ] = False,
):
    ckpt_file = get_best_model(
        pl_rootdir / f"version_{version}",
        metric,
        smaller_is_better=smaller_is_better,
    )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_file, output_dir / "checkpoints" / ckpt_file.name)
    for file in output_dir.iterdir():
        if file.suffix in [".json", ".csv", ".yaml", ".yml"]:
            shutil.copy2(file, output_dir / file.name)


if __name__ == "__main__":
    GlobalStorage.init(LIBACTOR_STORAGE_DIR)
    app()
