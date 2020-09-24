import os
import click

from dotenv import(
    load_dotenv, find_dotenv
)

import pytorch_lightning as pl

from lightning_module import LightningMNISTClassifier

@click.command()
@click.option(
    "--console / --no_console",
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
@click.option(
    "--batch_size",
    type=int,
    default=64,
    help="Number of examples per batch."
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate to use in training."
)
@click.option(
    "--fast_dev_run / --no_fast_dev_run",
    default=False,
    help="Enable to run a single batch for debugging."
)
def main(console, seed, epochs, batch_size, 
    learning_rate, fast_dev_run
):
    # Pytorch-lightning includes a seed_everything
    # function to set python, numpy, torch, etc, etc,
    # etc, for reproducability purposes.
    pl.seed_everything(seed)

    # Initialize the PL module
    model = LightningMNISTClassifier(
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Initialize the trainer.
    trainer = pl.Trainer(
        max_epochs=epochs, # Number of epochs to run.
        fast_dev_run=fast_dev_run, # Debug mode on if this flag is passed.
        val_percent_check=0.0, # Shuts off validation (not usually recommended!)
        num_sanity_val_steps=0, # We're not using a traditional validation set
        deterministic=True # Allows us to take advantage of the random seeds (incurs performance cost).
    )

    # Run the model with the trainer.
    trainer.fit(model)

    print(model.dot_prods)