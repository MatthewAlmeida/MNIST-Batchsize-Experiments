import os
import json
from dotenv import (
    load_dotenv, find_dotenv
)
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

import pytorch_lightning as pl

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

        self.mnist_dir = Path("/mnist")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_fold_size = train_fold_size
        self.pop_fold_grad = None
        self.train_fold_grad = None
        self.compare_grads = compare_grads
        self.dot_prods = []

        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
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

        self.pop_fold_grad = self.layer_3.weight.grad.clone()

        self.zero_grad()

    def training_step(self, train_batch, batch_idx):
        """
        Here we optionally sneak in a gradient computation
        for a population-level batch.
        """

        if self.compare_grads:
            self.populate_population_fold_grad()

        # Resume a normal training step here.
        X, y = train_batch
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def setup(self, stage_name):
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        mnist_train = MNIST(
            self.mnist_dir, train=True, download=True, transform=transform
        )
        self.mnist_test = MNIST(
            self.mnist_dir, train=False, download=True, transform=transform
        )

        """
        Split the usual MNIST training set into a small training set and a huge 
        validation set, meant to represent population-level statistics.
        """

        self.mnist_train, self.mnist_pop_fold = random_split(
            mnist_train, 
            [self.train_fold_size, len(mnist_train) - self.train_fold_size]
        )

        """
        Create the population-level dataloader and first iterator here.
        Further iterators are created as necessary during training in 
        populate_population_fold_grad.
        """

        self.population_dataloader = DataLoader(
            self.mnist_pop_fold, batch_size=self.batch_size, drop_last=True
        )

        self.population_iterator = iter(self.population_dataloader)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def compute_store_grad_dot_product(self):
        self.dot_prods.append(
            torch.dot(
                self.pop_fold_grad.flatten(), self.train_fold_grad.flatten()
            ).item()
        )

    def on_after_backward(self):
        if self.compare_grads:
            self.train_fold_grad = self.layer_3.weight.grad.clone()
            self.compute_store_grad_dot_product()

