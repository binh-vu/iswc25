from __future__ import annotations

from ast import Global
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import lightning.pytorch as pl
import serde.json
import wandb
from experiments.config import DATA_DIR, LIBACTOR_STORAGE_DIR
from libactor.storage import GlobalStorage
from loguru import logger
from pytorch_lightning.loggers import CSVLogger, WandbLogger, tensorboard
from sm.misc.funcs import assert_isinstance, get_incremental_path

from gpp.sem_label.data_modules.base import get_best_model
from gpp.sem_label.data_modules.v210 import SLabelV210DataModule
from gpp.sem_label.feats import get_text_embedding
from gpp_scripts.sem_label.contrastive.train.ablation_study.v221a_without_column_header import (
    V221a,
)

enable_wandb = False
DATASET_DIR = DATA_DIR / "experiments/sem-label-data-v210"
print("Set data directory:", DATA_DIR)


def v221a_exp(
    embedding_model: str,
    embedding_model_customization: str,
    compress_dim: int,
    margin: float,
    sim_metric: Literal["cosine", "vecnorm"],
    norm_input_embedding: bool,
    restore_version: Optional[int] = None,
    max_epochs: int = 5,
):
    datamodule = SLabelV210DataModule(
        DATASET_DIR,
        embedding_model,
        get_text_embedding(embedding_model, embedding_model_customization),
        train_batch_size=32,
        eval_batch_size=32,
    )
    datamodule.setup("fit")

    L.seed_everything(42, workers=True)
    assert isinstance(datamodule.dev, list)
    model = V221a(
        embedding_dim=datamodule.get_embedding_dim(),
        compress_dim=compress_dim,
        margin=margin,
        sim_metric=sim_metric,
        norm_input_embedding=norm_input_embedding,
        val_dataloader_names=[d.name for d in datamodule.dev],
    )
    model.set_references(datamodule)

    pl_rootdir = DATA_DIR / "lightning"
    pl_logger = [
        CSVLogger(pl_rootdir / "csv", name="sem-label"),
        tensorboard.TensorBoardLogger(pl_rootdir / "tensorboard", name="sem-label"),
    ]
    enable_wandb = False

    if enable_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        pl_logger.append(
            WandbLogger(
                project="grams++sem-label", save_dir=pl_rootdir, log_model="all"
            )
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # type: ignore
        save_top_k=100,
        monitor=f"{model.valsets[0]}_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=pl_logger,
        default_root_dir=pl_rootdir,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    if restore_version is None:
        try:
            trainer.fit(model=model, datamodule=datamodule)
        except:
            if enable_wandb:
                wandb.finish(exit_code=1)
            raise
        finally:
            version = assert_isinstance(trainer.logger, CSVLogger).version
            serde.json.ser(
                datamodule.params,
                (
                    Path(assert_isinstance(trainer.logger, CSVLogger).root_dir)
                    / f"version_{version}"
                    / "datamodule_params.json"
                ),
                indent=2,
            )
    else:
        version = restore_version

    log_dir = (
        Path(assert_isinstance(trainer.logger, CSVLogger).root_dir)
        / f"version_{version}"
    )
    metric = "val_t2dv2_mrr"

    ckpt_file = get_best_model(log_dir, metric, smaller_is_better=False)
    best_model = model.__class__.load_from_checkpoint(ckpt_file)
    logger.info("Best model according to {} is at {}", metric, ckpt_file)

    model = best_model
    model.set_references(datamodule)

    for dl in datamodule.val_dataloader():
        trainer.validate(model, dl)
        model.evaluate(dl)


if __name__ == "__main__":
    GlobalStorage.init(LIBACTOR_STORAGE_DIR)
    v221a_exp(
        embedding_model="BAAI/bge-m3",
        embedding_model_customization="default",
        compress_dim=256,
        margin=0.10,
        sim_metric="cosine",
        norm_input_embedding=True,
        max_epochs=5,
    )
