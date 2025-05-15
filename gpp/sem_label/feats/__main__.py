from __future__ import annotations

import os
from typing import Annotated

import typer
from keyvec import HfModel, HfModelArgs
from slugify import slugify
from sm.misc.ray_helper import RemoteService

app = typer.Typer()


@app.command(help="Serve a HF model using Ray.")
def start(
    model: str,
    num_replicas: Annotated[int, typer.Option()] = 1,
    num_cpus: Annotated[int, typer.Option()] = 4,
    num_gpus: Annotated[int, typer.Option()] = 0,
    port: Annotated[int, typer.Option()] = 18861,
    blocking: Annotated[bool, typer.Option("--blocking")] = False,
):
    if "HF_REMOTE" in os.environ:
        del os.environ["HF_REMOTE"]

    from ray import serve

    # it's safe to run this multiple times as ray will ignore it if it's already running
    # on the ray cluster
    serve.start(http_options={"host": "0.0.0.0", "port": port})

    ray_actor_options = {}
    if num_gpus > 0:
        ray_actor_options["num_gpus"] = num_gpus
    if num_cpus > 0:
        ray_actor_options["num_cpus"] = num_cpus

    RemoteService.start(
        HfModel,
        (HfModelArgs(model),),
        {
            "num_replicas": num_replicas,
            "ray_actor_options": ray_actor_options,
        },
        name=slugify(model),
        blocking=blocking,
    )


if __name__ == "__main__":
    app()
