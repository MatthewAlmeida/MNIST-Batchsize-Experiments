import os
import click

from dotenv import(
    load_dotenv, find_dotenv
)

import pytorch_lightning as pl

from lightning_module import LightningMNISTClassifier

@click.command()
@click.option(
    "--console / --no-cosole",
    default=True,
    help="Pass --console to disable database and output to console."
)
@click.option(
    "--seed",
    type=int,
    default=55,
    help="The value to set PL seed_everything function with."
)
@click.option(
    "--epochs",
    type=int,
    default=1,
    help="Maximum number of epochs to train for."
)
def main(console, seed, epochs):
    # train
    model = LightningMNISTClassifier()
    trainer = pl.Trainer(
        max_epochs = epochs,
        deterministic = True
    )

    pl.seed_everything(seed)
    trainer.fit(model)