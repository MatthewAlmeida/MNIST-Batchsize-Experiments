import os
import click

from dotenv import(
    load_dotenv, find_dotenv
)
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

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
    "--train_fold_size",
    type=int,
    default=5000,
    help=f"Number of training examples to use. 60000-[this number] will "
          "be used for validation / population gradient checks."
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
    train_fold_size, learning_rate, fast_dev_run
):
    # Make sure .env variables are loaded.
    load_dotenv(find_dotenv())

    # Get the log dir variable; default to /logs.
    # Create if not exists.
    log_dir = Path(f"/{os.getenv('LOG_DIR')}" or "/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pytorch-lightning includes a seed_everything
    # function to set python, numpy, torch, etc, etc,
    # etc, for reproducability purposes.
    pl.seed_everything(seed)

    # Initialize tensorbaord logger
    tb_logger = TensorBoardLogger(log_dir)
    click.echo(f"Initialized Tensorboard logger to log dir {log_dir}")

    # Initialize the PL module
    model = LightningMNISTClassifier(
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_fold_size=train_fold_size
    )

    # Initialize the trainer.
    trainer = pl.Trainer(
        max_epochs=epochs, # Number of epochs to run.
        fast_dev_run=fast_dev_run, # Debug mode on if this flag is passed.
        limit_val_batches=150,
        num_sanity_val_steps=5, 
        deterministic=True, # Allows us to take advantage of the random seeds (incurs performance cost).
        logger=tb_logger
    )

    # Run the model with the trainer.
    trainer.fit(model)
