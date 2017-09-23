
import os

import torch
import torch.nn as nn

from seqmod.modules.lm import LM
from seqmod.misc.dataset import BlockDataset, Dict, CompressionTable
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.early_stopping import EarlyStopping
from seqmod.misc.trainer import CLMTrainer
from seqmod.misc.loggers import StdLogger
import seqmod.utils as u

from utils import readlines


def linearize_data(lines, conds, lang_d, conds_d, table=None):
    for line, line_conds in zip(lines, conds):
        line_conds = tuple(d.index(c) for d, c in zip(conds_d, line_conds))
        for char in next(lang_d.transform([line])):
            yield char
            if table is None:
                for c in line_conds:
                    yield c
            else:
                yield table.hash_vals(line_conds)


def examples_from_lines(lines, conds, lang_d, conds_d, table=None):
    t = linearize_data(lines, conds, lang_d, conds_d, table=table)
    t = torch.LongTensor(list(t))
    if table is not None:       # text + encoded conditions
        return t.view(-1, 2).t().contiguous()
    else:                       # text + conditions
        return t.view(-1, len(conds_d) + 1).t().contiguous()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=48, type=int)
    parser.add_argument('--cond_emb_dim', default=48)
    parser.add_argument('--hid_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--deepout_layers', default=1, type=int)
    parser.add_argument('--maxouts', default=3, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    # dataset
    parser.add_argument('--path')
    # training
    parser.add_argument('--epochs', default=75, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--bptt', default=50, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--dev_split', type=float, default=0.1)
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    # - check
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    args = parser.parse_args()

    models_dir = 'models/clm'
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    # dataset
    print("Loading data")
    conds, lines = [], []
    for f in os.listdir(args.path):
        for cs, line in readlines(os.path.join(args.path, f)):
            if len(cs) == 0:
                continue
            conds.append(cs), lines.append(line)
    conds_d = [Dict(sequential=False, force_unk=False)
               for _ in range(len(conds[0]))]
    lang_d = Dict(eos_token=u.EOS)
    print("Fitting language Dict")
    lang_d.fit(lines)
    print(lang_d)
    print("Fitting condition Dicts")
    for d, cond in zip(conds_d, list(map(list, zip(*conds)))):
        d.fit([cond])
    table = CompressionTable(len(conds[0]))
    data = examples_from_lines(lines, conds, lang_d, conds_d, table=table)
    del lines, conds
    d = tuple([lang_d] + conds_d)
    train, valid = BlockDataset.splits_from_data(
        tuple(data), d, args.batch_size, args.bptt, gpu=args.gpu,
        dev=None, test=args.dev_split, table=table)

    # model
    print("Building model")
    conds = []
    for idx, subd in enumerate(conds_d):
        print(' * condition [{}] with cardinality {}'.format(idx, len(subd)))
        conds.append({'varnum': len(subd), 'emb_dim': args.cond_emb_dim})

    m = LM(len(lang_d), args.emb_dim, args.hid_dim,
           num_layers=args.layers, cell=args.cell, dropout=args.dropout,
           deepout_layers=args.deepout_layers, deepout_act=args.deepout_act,
           conds=conds)
    u.initialize_model(m)
    if args.gpu:
        m.cuda()

    # trainer
    optim = Optimizer(
        m.parameters(), args.optim, lr=args.lr, max_norm=args.max_norm)
    crit = nn.NLLLoss()
    early_stopping = EarlyStopping(
        10, patience=args.patience, reset_patience=False)
    trainer = CLMTrainer(m, {"train": train, "valid": valid}, crit, optim,
                         early_stopping=early_stopping)
    logger = StdLogger(os.path.join(args.path, f'clm.train'))
    trainer.add_loggers(logger)

    # run
    (best_m, valid_loss), _ = trainer.train(
        args.epochs, args.checkpoint, gpu=args.gpu)

    m_path = os.path.join(models_dir, f'clm_{valid_loss}')
    u.save_model(best_m, m_path, d=(d, table))
