#!/usr/bin/env python3
#
# File: delan.py
#

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
from datetime import datetime
from pathlib import Path

import click
import torch

from mechamodlearn import dataset, utils, viz_utils
from mechamodlearn.trainer import SimpleDeLaNTrainer
from mechamodlearn.systems import MultiLinkAcrobot
from mechamodlearn.rigidbody import LearnedRigidBody, DeLaN
from mechamodlearn.models import CholeskyMMNet, SymmetricMMNet

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

system_maker = lambda p: MultiLinkAcrobot(2, p)


def get_datagen(system, batch_size: int, fscale: float, qrange=(-1, 1),
                vrange=(-10, 10)):
    while True:
        q = torch.stack([torch.empty(system._qdim).uniform_(*qrange)
                         for _ in range(batch_size)]).requires_grad_(True)
        v = torch.stack([torch.empty(system._qdim).uniform_(*vrange)
                         for _ in range(batch_size)]).requires_grad_(True)
        F = torch.randn(batch_size, system._qdim) * fscale
        qddot = system.solve_euler_lagrange_from_F(q, v, F)
        # shape of each element: (batch, dim)
        yield (q, v, qddot, F)


def train(seed, mm, lr, n_batches, batch_size, fscale, logdir):
    args = locals()
    args.pop('logdir')
    args.pop('n_batches')
    exp_name = ",".join(["=".join([key, str(val)]) for key, val in args.items()]) + time.strftime('%Y%m%d%H%M')

    utils.set_rng_seed(seed)

    system = system_maker(torch.tensor([10.] * 2            # mass
                                       + [1.] * 2           # length
                                       + [10.]              # g
                                       + [1.] * 2           # gain of input
                                      ))
    
    datagen = get_datagen(system, batch_size, fscale)

    # use 1 batch for validation
    valid_dataset = next(datagen)

    if mm == 'chol':
        mass_matrix_type = CholeskyMMNet
    elif mm == 'chol_relu':
        mass_matrix_type = lambda *args, **kwargs: CholeskyMMNet(*args, **kwargs, bias=0,
                                                                 pos_enforce=torch.nn.functional.relu)
    elif mm == 'symm':
        mass_matrix_type = SymmetricMMNet
    model = DeLaN(system._qdim, system.thetamask, mass_matrix_type,
                             hidden_sizes=[30, 30, 30, 30])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if logdir is not None:
        logdir = Path(logdir) / exp_name

    trainer = SimpleDeLaNTrainer(model, system, opt, datagen, valid_dataset,
                                 num_train_batches=n_batches, batch_size=batch_size,
                                 logdir=logdir, device=DEVICE)

    metrics = trainer.train()

    if logdir is not None:
        torch.save(metrics, Path(logdir) / 'metrics_{:%Y%m%d-%H%M%S}.pt'.format(datetime.now()))
    return metrics


@click.command()
@click.option('--seed', default=42, type=int)
@click.option('--mm', default='chol', type=str)
@click.option('--lr', default=1e-3, type=float)
@click.option('--n-batches', default=10000, type=int)
@click.option('--batch-size', default=256, type=int)
@click.option('--fscale', default=30.0, type=float)
@click.option('--logdir', default=None, type=str)
def run(seed, mm, lr, n_batches, batch_size, fscale, logdir):
    metrics = train(seed, mm, lr, n_batches, batch_size, fscale,
                    logdir)
    print(metrics)


if __name__ == '__main__':
    run()