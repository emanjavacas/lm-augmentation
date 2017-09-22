
import os
import warnings

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    warnings.warn('no NVIDIA driver found')
    torch.manual_seed(1001)

import torch.nn as nn

from seqmod.modules.lm import LM
from seqmod import utils as u

from seqmod.misc.trainer import LMTrainer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.early_stopping import EarlyStopping

from utils import readlines, get_author


def load_lines(path, author):
    lines = []
    for f in os.listdir(path):
        if not f.startswith(author):
            continue
        for _, l in readlines(os.path.join(path, f)):
            lines.append(l.strip())
    return lines


def load_dataset(path, author, args):
    d, data = Dict(eos_token=u.EOS), load_lines(path, author)
    d.fit(data)
    train, valid = BlockDataset(
        data, d, args.batch_size, args.bptt, gpu=args.gpu
    ).splits(test=args.dev_split, dev=None)
    return train, valid, d


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    parser.add_argument('--maxouts', default=2, type=int)
    parser.add_argument('--train_init', action='store_true')
    # dataset
    parser.add_argument('--path', required=True)
    parser.add_argument('--dev_split', default=0.1, type=float)
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--early_stopping', default=3, type=int)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    args = parser.parse_args()

    authors = [get_author(f) for f in os.listdir(args.path)]
    models_dir = 'models/lm'
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    for author in authors:
        print(f"Training generator for {author}")

        # dataset
        train, valid, d = load_dataset(args.path, author, args)

        # model
        m = LM(len(d), args.emb_dim, args.hid_dim,
               num_layers=args.num_layers, cell=args.cell,
               dropout=args.dropout, train_init=args.train_init,
               deepout_layers=args.deepout_layers,
               deepout_act=args.deepout_act, maxouts=args.maxouts)
        u.initialize_model(m)
        if args.gpu:
            m.cuda()

        # trainer
        optim = Optimizer(
            m.parameters(), args.optim, lr=args.lr, max_norm=args.max_norm)
        crit = nn.NLLLoss()
        early_stopping = EarlyStopping(10, patience=args.early_stopping)
        trainer = LMTrainer(m, {"train": train, "valid": valid}, crit, optim,
                            early_stopping=early_stopping)
        logger = StdLogger(os.path.join(args.path, f'{author}.train'))
        trainer.add_loggers(logger)

        # run
        (best_m, valid_loss), _ = trainer.train(
            args.epochs, args.checkpoint, gpu=args.gpu)

        m_path = os.path.join(models_dir, f'{author}_{valid_loss}')
        u.save_model(best_m, m_path, d=d)
