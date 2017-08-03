
import os
import math
from pprint import pprint

from seqmod import utils as u
from seqmod.loaders import load_penn3
from seqmod.misc.dataset import BlockDataset, Dict
from seqmod.misc.early_stopping import EarlyStopping
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger
from seqmod.hyper import make_sampler, Hyperband

import pos_aware_lm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path')
    parser.add_argument('--dataset_path')
    parser.add_argument('--load_dataset', action='store_true')
    parser.add_argument('--save_dataset', action='store_true')
    parser.add_argument('--max_size', default=100000, type=int)
    # training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bptt', type=int, default=30)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.load_dataset:
        dataset = BlockDataset.from_disk(args.dataset_path)
    else:
        words, pos = zip(*load_penn3(args.path, swbd=False))
        word_dict = Dict(
            eos_token=u.EOS, bos_token=u.BOS, force_unk=True,
            max_size=args.max_size)
        pos_dict = Dict(
            eos_token=u.EOS, bos_token=u.BOS, force_unk=False)
        word_dict.fit(words), pos_dict.fit(pos)
        dataset = BlockDataset(
            (pos, words), (pos_dict, word_dict), args.batch_size, args.bptt)
        if args.save_dataset and not os.path.isfile(args.dataset_path):
            dataset.to_disk(args.dataset_path)

    pos_dict, word_dict = dataset.d

    std_logger = StdLogger()

    def try_params(epochs, params):
        epochs = int(epochs)

        m = pos_aware_lm.ChainPOSAwareLM(
            (len(pos_dict.vocab), len(word_dict.vocab)),  # vocabs
            (params['pos_emb_dim'], params['word_emb_dim']),
            (params['pos_hid_dim'], params['word_hid_dim']),
            num_layers=(params['pos_num_layers'], params['word_num_layers']),
            dropout=params['dropout'])

        print(m)

        dataset.set_gpu(args.gpu)
        train, valid = dataset.splits(test=None)

        m.apply(u.make_initializer())
        if args.gpu:
            m.cuda()

        early_stopping = EarlyStopping(10, patience=3, reset_patience=False)
        crit = pos_aware_lm.make_pos_word_criterion(gpu=args.gpu)
        optim = Optimizer(m.parameters(), args.optim, lr=params['lr'])
        trainer = pos_aware_lm.POSAwareLMTrainer(
            m, {'train': train, 'valid': valid}, crit, optim,
            pos_weight=params['pos_weight'],
            early_stopping=early_stopping)
        trainer.add_loggers(std_logger)
        (best_model, loss), _ = trainer.train(epochs, args.checkpoint)

        return {'loss': loss, 'log_loss': loss, 'early_stop': early_stopping.stopped}

    get_params = make_sampler({
        'pos_emb_dim': ['choice', int, (24, 48)],
        'pos_hid_dim': ['choice', int, (200, 400, 600)],
        'word_emb_dim': ['choice', int, (24, 48)],
        'word_hid_dim': ['choice', int, (200, 400, 600)],
        'pos_num_layers': ['choice', int, (1, 2)],
        'word_num_layers': ['choice', int, (1, 2)],
        'dropout': ['loguniform', float, math.log(0.1), math.log(0.5)],
        'pos_weight': ['loguniform', float, math.log(0.2), math.log(0.8)],
        'lr': ['loguniform', float, math.log(0.001), math.log(0.05)]
    })

    pprint(Hyperband(get_params, try_params).run())
