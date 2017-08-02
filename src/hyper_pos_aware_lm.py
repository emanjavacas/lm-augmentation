
from pprint import pprint

from seqmod import utils as u
from seqmod.misc.dataset import BlockDataset
from seqmod.misc.early_stopping import EarlyStopping
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger
from seqmod.hyper import make_sampler, Hyperband

import pos_aware_lm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoints', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    dataset = BlockDataset.from_disk(args.dataset_path)

    pos_dict, word_dict = dataset.d

    std_logger = StdLogger()

    def try_params(params):
        m = pos_aware_lm.ChainPOSAwareLM(
            (len(pos_dict.vocab), len(word_dict.vocab)),  # vocabs
            (params['pos_emb_dim'], params['word_emb_dim']),
            (params['pos_hid_dim'], params['word_hid_dim']),
            num_layers=(params['pos_num_layers'], params['word_num_layers']),
            dropout=params['dropout'])

        dataset.set_batch_size(params['batch_size']), dataset.set_gpu(args.gpu)
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
        (best_model, loss), _ = trainer.train(args.epochs, args.checkpoint)

        return {'loss': loss, 'log_loss': loss, 'early_stop': early_stopping.stopped}

    get_params = make_sampler({
        'pos_emb_dim': [],
        'pos_hid_dim': [],
        'word_emb_dim': [],
        'word_hid_dim': [],
        'pos_num_layers': [],
        'word_num_layers': [],
        'dropout': [],
        'batch_size': [],
        'pos_weight': []
    })

    pprint(Hyperband(get_params, try_params).run())
