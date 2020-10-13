import os
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

"""
Based on code provided in the following Medium article by William Falcon:
https://towardsdatascience.com/
    from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
"""

class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, 
        batch_size=64, learning_rate=0.001,
        train_fold_size = 55000, compare_grads = True
    ):
        super(LightningMNISTClassifier, self).__init__()

        # Look for the dataset in the directory specified in the 
        # environment variable. If it doesn't exist, download the 
        # dataset (very fast) to /mnist.

        self.mnist_dir = (Path(f"/{os.getenv('MNIST_DATA_DIR')}") 
            or Path("/mnist")
        )

        # Define training hyperparams and necessary variables.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_fold_size = train_fold_size
        self.compare_grads = compare_grads
        self.epoch_idx = 0

        # Define net components.

        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        """
        Define forward propagation behavior. We're using a
        *very* simple network for this demonstration.
        """

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        l3_out = self.layer_3(x)

        # probability distribution over labels
        out = torch.log_softmax(l3_out, dim=1)

        return out

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def setup(self, stage_name):
        """
        This function is called by pytorch-lightning before training
        to set up needed state.
        """

        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        mnist_train = MNIST(
            self.mnist_dir, train=True, download=True, transform=transform
        )
        self.mnist_test = MNIST(
            self.mnist_dir, train=False, download=True, transform=transform
        )

        # Split the usual MNIST training set into a small training set and a 
        # huge validation set, meant to represent population-level statistics.
        
        self.mnist_train, self.mnist_pop_fold = random_split(
            mnist_train, 
            [self.train_fold_size, len(mnist_train) - self.train_fold_size]
        )

        # Create the population-level dataloader and first iterator here.
        # Further iterators are created as necessary during training in 
        # populate_population_fold_grad.

        self.population_dataloader = DataLoader(
            self.mnist_pop_fold, batch_size=self.batch_size, drop_last=True
        )
        self.population_iterator = iter(self.population_dataloader)

        self.logger.log_hyperparams(
            {
                "Batch size": self.batch_size,
                "Learning rate": self.learning_rate,
                "Train fold size": self.train_fold_size
            }
        )

        # Initialize tensors to hold histograms of dot products
        # for weights and biases.
        self.W_dot_prods = torch.zeros(self.train_fold_size)
        self.B_dot_prods = torch.zeros(self.train_fold_size)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    """
    We are intentionally using the test set as a validation set here,
    because we don't actually care much about the validation / test
    performance - we're concerned with measuring an aspect of training. We
    also support the test_dataloader function so that test functions on 
    this model class run as expected, even though it's the same as the 
    validation set.
    """
    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def populate_population_fold_grad(self):
        """
        Draw a population-level batch, store the gradient 
        of the last layer, then zero the grads again.
        As the training fold and population fold can
        be different sizes, their iterators may run out of 
        data at different times. If that happens, we 
        make a new iterator and continue on. This does
        shuffle the dataset.
        """
        try:
            X, y = next(self.population_iterator)
        except StopIteration:
            self.population_iterator = iter(self.population_dataloader)
            X, y = next(self.population_iterator)

        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)
        loss.backward()

        self.pop_fold_W_grad = self.layer_3.weight.grad.clone()
        self.pop_fold_B_grad = self.layer_3.bias.grad.clone()

        self.zero_grad()

    def training_step(self, train_batch, batch_idx):
        """
        Training code. We optionally sneak in a gradient 
        computation for a population-level batch. Store
        batch index to make it available to hooks that
        don't get it from the trainer (on_after_backward).
        """
        self.batch_idx = batch_idx

        if self.compare_grads:
            self.populate_population_fold_grad()

        # Resume a normal training step here.
        X, y = train_batch
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        result = pl.TrainResult(minimize = loss)
        result.log('train_loss', loss, prog_bar=True)

        return result

    def validation_step(self, valid_batch, batch_idx):
        X, y = valid_batch

        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        result = pl.EvalResult(checkpoint_on=loss)

        result.log('val_loss', loss, prog_bar=True)
        result.log('val_acc', acc, prog_bar=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def compute_store_grad_dot_product(self):
        """
        Here we calculate the dot product between the training 
        and validation gradients and store the value in the dot_prods
        list.
        """

        self.W_dot_prods[self.batch_idx] = torch.dot(
            self.pop_fold_W_grad.flatten(), self.train_fold_W_grad.flatten()
        )
        self.B_dot_prods[self.batch_idx] = torch.dot(
            self.pop_fold_B_grad.flatten(), self.train_fold_B_grad.flatten()
        )

    def on_after_backward(self):
        """
        Pytorch-lightning invokes this callback after the loss is 
        backpropagated but before the optimizers change the 
        model parameters. We calculate and store the dot products
        here.
        """

        if self.compare_grads:
            self.train_fold_W_grad = self.layer_3.weight.grad.clone()
            self.train_fold_B_grad = self.layer_3.bias.grad.clone()
            self.compute_store_grad_dot_product()

    def on_epoch_end(self):
        # Loggers of other types might not have experiment 
        # objects or add_histogram methods; if not, ignore.
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_histogram(
                "Weight Gradient Dot Products", self.W_dot_prods, 
                self.epoch_idx
            )
            self.logger.experiment.add_histogram(
                "Bias Gradient Dot Products", self.B_dot_prods,
                self.epoch_idx
            )

        # Each of these tensors hold one epoch's worth of 
        # dot products; at epoch end, re-initialize them to 
        # zeros.
        self.W_dot_prods.fill_(0.0)
        self.B_dot_prods.fill_(0.0)
        self.epoch_idx += 1
