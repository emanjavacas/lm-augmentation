
import os
import math

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod import utils as u

from seqmod.modules.custom import StackedGRU, StackedLSTM
from seqmod.loaders import load_penn3
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger


class ChainPOSAwareLM(nn.Module):
    """
    Word-level LM conditioned on a pos-level model.

    This model implements pos-conditioning using the last RNN hidden-layer
    activations. Conditioning is done by concatenation with the current input
    embedding. Conditioning is done in both ways (i.e. the last hidden
    activation of the word-level RNN is concatenated to the current
    POS-embedding), the difference being that at a given step t the POS-RNN
    is conditioned by the last word-RNN layer activation at time t-1, and
    the word-RNN is conditioned by the last POS-RNN layer activation at
    time step t. This visually resembles some kind of chain, hence the name.
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers,
                 dropout=0.0, cell='LSTM',
                 tie_pos=False, tie_word=True,
                 pos_gate=True, word_gate=True):
        self.pos_vocab, self.word_vocab = vocab
        self.pos_emb_dim, self.word_emb_dim = emb_dim
        self.pos_hid_dim, self.word_hid_dim = hid_dim
        self.pos_num_layers, self.word_num_layers = num_layers
        self.pos_gate, self.word_gate = pos_gate, word_gate
        self.cell = cell
        self.dropout = dropout
        super(ChainPOSAwareLM, self).__init__()

        # embeddings
        self.pos_emb = nn.Embedding(self.pos_vocab, self.pos_emb_dim)
        self.word_emb = nn.Embedding(self.word_vocab, self.word_emb_dim)

        stacked = StackedLSTM if self.cell == 'LSTM' else StackedGRU
        # pos network
        # pos rnn
        self.pos_rnn = stacked(
            self.pos_num_layers,
            self.pos_emb_dim + self.word_hid_dim,
            self.pos_hid_dim,
            dropout=self.dropout)
        # pos gate
        if pos_gate:
            self.p2w_gate = nn.Sequential(
                nn.Linear(self.pos_hid_dim, self.pos_hid_dim), nn.Tanh())
        # pos output projection
        if tie_pos:
            pos_project = nn.Linear(self.pos_emb_dim, self.pos_vocab)
            pos_project.weight = self.pos_emb.weight
            self.pos_project = nn.Sequential(
                nn.Linear(self.pos_hid_dim, self.pos_emb_dim),
                pos_project,
                nn.LogSoftmax())
        else:
            self.pos_project = nn.Sequential(
                nn.Linear(self.pos_hid_dim, self.pos_vocab),
                nn.LogSoftmax())

        # word network
        # word rnn
        self.word_rnn = stacked(
            self.word_num_layers,
            self.word_emb_dim + self.pos_hid_dim,
            self.word_hid_dim,
            dropout=self.dropout)
        # word gate
        if word_gate:
            self.w2p_gate = nn.Sequential(
                nn.Linear(self.word_hid_dim, self.word_hid_dim), nn.Tanh())
        # word output projection
        if tie_word:
            word_project = nn.Linear(self.word_emb_dim, self.word_vocab)
            word_project.weight = self.word_emb.weight
            self.word_project = nn.Sequential(
                nn.Linear(self.word_hid_dim, self.word_emb_dim),
                word_project,
                nn.LogSoftmax())
        else:
            self.word_project = nn.Sequential(
                nn.Linear(self.word_hid_dim, self.word_vocab),
                nn.LogSoftmax())

    def init_hidden_for(self, inp, source_type):
        batch = inp.size(0)
        if source_type == 'pos':
            size = (self.pos_num_layers, batch, self.pos_hid_dim)
        else:
            assert source_type == 'word'
            size = (self.word_num_layers, batch, self.word_hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def get_last_hid(self, h):
        if self.cell.startswith('LSTM'):
            h, _ = h
        return h[-1]

    def step(self, p, w, p_hid, w_hid):
        """
        p: tensor (batch_size x emb_dim), embedding for POS-tag at current step
        w: tensor (batch_size x emb_dim), embedding for word at current step
        p_hid: tensor (num_layers x batch_size x hid_dim)
        w_hid: tensor (num_layers x batch_size x hid_dim)
        """
        last_w_hid = self.get_last_hid(w_hid)
        if self.word_gate:
            last_w_hid = self.w2p_gate(last_w_hid)
        p_out, p_hid = self.pos_rnn(
            torch.cat([p, last_w_hid], 1),
            hidden=p_hid)
        p_out = self.pos_project(p_out)
        last_p_hid = self.get_last_hid(p_hid)
        if self.pos_gate:
            last_p_hid = self.p2w_gate(last_p_hid)
        w_out, w_hid = self.word_rnn(
            torch.cat([w, last_p_hid], 1),
            hidden=w_hid)
        w_out = self.word_project(w_out)
        return (p_out, w_out), (p_hid, w_hid)

    def forward(self, pos, word, hidden=None):
        """
        <bos>/<bos> NNP/Pierre NNP/Vinken CD/61 ... ,/, MD/will <eos>/<eos>

        ==Input==
              1 (pos/word) 2          3                n-1
        POS:  <bos>/<bos>  NNP/Pierre NNP/Vinken  ...  MD/will
        word: NNP/<bos>    NNP/Pierre CD/Vinken   ...  <eos>/will

        ==Output==
              1            2          3                n-1
        POS:  NNP          NNP        CD          ...  <eos>
        word: Pierre       Vinken     61          ...  <eos>
        """
        p_outs, w_outs = [], []
        p_hid, w_hid = hidden if hidden is not None else (None, None)
        p_emb, w_emb = self.pos_emb(pos), self.word_emb(word)
        for p, w in zip(p_emb, w_emb):
            p_hid = p_hid or self.init_hidden_for(p, 'pos')
            w_hid = w_hid or self.init_hidden_for(w, 'word')
            (p_out, w_out), (p_hid, w_hid) = self.step(
                p, w, p_hid=p_hid, w_hid=w_hid)
            p_outs.append(p_out), w_outs.append(w_out)
        return (torch.stack(p_outs), torch.stack(w_outs)), (p_hid, w_hid)

    def generate(self, p_dict, w_dict, seed=None, max_seq_len=20, hidden=None,
                 temperature=1., batch_size=5, gpu=False, ignore_eos=False):

        def sample(out):
            prev = out.div(temperature).exp_().multinomial().t()
            score = u.select_cols(out.data.cpu(), prev.squeeze().data.cpu())
            return prev, score

        def init_prev(bos):
            out = Variable(torch.LongTensor([bos] * batch_size), volatile=True)
            out = out.cuda() if gpu else out
            return out

        finished = np.array([False] * batch_size)
        p_hyp, w_hyp, p_scores, w_scores = [], [], 0, 0
        p_hid, w_hid = hidden if hidden is not None else (None, None)
        w_eos = word_dict.get_eos()
        p_prev = init_prev(pos_dict.get_bos())
        w_prev = init_prev(word_dict.get_bos())
        for _ in range(max_seq_len):
            p_emb, w_emb = self.pos_emb(p_prev), self.word_emb(w_prev)
            p_hid = p_hid or self.init_hidden_for(p_emb, 'pos')
            w_hid = w_hid or self.init_hidden_for(w_emb, 'word')
            (p_out, w_out), (p_hid, w_hid) = self.step(
                p_emb, w_emb, p_hid=p_hid, w_hid=w_hid)
            (p_prev, p_score), (w_prev, w_score) = sample(p_out), sample(w_out)
            # hyps
            mask = (w_prev.squeeze().data == w_eos).cpu().numpy() == 1
            finished[mask] = True
            if all(finished == True): break
            p_hyp.append(p_prev.squeeze().data.tolist())
            w_hyp.append(w_prev.squeeze().data.tolist())
            # scores
            p_score[torch.ByteTensor(finished.tolist())] = 0
            w_score[torch.ByteTensor(finished.tolist())] = 0
            p_scores, w_scores = p_scores + p_score, w_scores + w_score

        return (list(zip(*p_hyp)), list(zip(*w_hyp))), \
            (p_score.tolist(), w_score.tolist())


def repackage_hidden(hidden):
    def _repackage_hidden(h):
        if isinstance(h, tuple):
            return tuple(_repackage_hidden(v) for v in h)
        else:
            return Variable(h.data)
    p_hid, w_hid = hidden
    return (_repackage_hidden(p_hid), _repackage_hidden(w_hid))


class POSAwareLMTrainer(Trainer):
    def __init__(self, *args, pos_weight=0.5, **kwargs):
        super(POSAwareLMTrainer, self).__init__(*args, **kwargs)
        if pos_weight < 0 or pos_weight > 1:
            raise ValueError("pos_weight must be between 0 and 1")
        self.pos_weight = pos_weight
        self.loss_labels = ('pos', 'word')

    def format_loss(self, losses):
        return tuple(math.exp(min(loss, 100)) for loss in losses)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        (src_pos, src_word), (trg_pos, trg_word) = batch_data
        seq_len, batch_size = src_pos.size()
        hidden = self.batch_state.get('hidden')
        (p_out, w_out), hidden = self.model(src_pos, src_word, hidden=hidden)
        self.batch_state['hidden'] = repackage_hidden(hidden)
        p_loss, w_loss = self.criterion(
            p_out.view(seq_len * batch_size, -1), trg_pos.view(-1),
            w_out.view(seq_len * batch_size, -1), trg_word.view(-1))
        if dataset == 'train':
            weighted_p_loss = p_loss * self.pos_weight
            weighted_w_loss = w_loss * (1 - self.pos_weight)
            (weighted_p_loss + weighted_w_loss).backward()
            self.optimizer_step()
        return p_loss.data[0], w_loss.data[0]

    def num_batch_examples(self, batch_data):
        (pos_src, _), _ = batch_data
        return pos_src.nelement()


def make_pos_word_criterion(gpu=False):
    p_crit, w_crit = nn.NLLLoss(), nn.NLLLoss()
    if gpu:
        p_crit.cuda(), w_crit.cuda()

    def criterion(p_outs, p_targets, w_outs, w_targets):
        return p_crit(p_outs, p_targets), w_crit(w_outs, w_targets)

    return criterion


def hyp_to_str(p_hyp, w_hyp, pos_dict, word_dict):
    p_str, w_str = "", ""
    for p, w in zip(p_hyp, w_hyp):
        p = pos_dict.vocab[p]
        w = word_dict.vocab[w]
        ljust = max(len(p), len(w)) + 2
        p_str += p.ljust(ljust, ' ')
        w_str += w.ljust(ljust, ' ')
    return p_str, w_str


def make_generate_hook(pos_dict, word_dict):
    def hook(trainer, epoch, batch, checkpoints):
        (p_hyps, w_hyps), (p_scores, w_scores) = \
            trainer.model.generate(pos_dict, word_dict, gpu=args.gpu)
        for p, w, p_score, w_score in zip(p_hyps, w_hyps, p_scores, w_scores):
            p_str, w_str = hyp_to_str(p, w, pos_dict, word_dict)
        trainer.log("info", "Score [%g, %g]: \n%s\n%s" %
                    (p_score, w_score, p_str, w_str))
    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', default='/home/enrique/corpora/penn3/')
    parser.add_argument('--dataset_path')
    parser.add_argument('--load_dataset', action='store_true')
    parser.add_argument('--save_dataset', action='store_true')
    parser.add_argument('--max_size', default=1000000, type=int)
    # model
    parser.add_argument('--pos_emb_dim', default=24, type=int)
    parser.add_argument('--word_emb_dim', default=64, type=int)
    parser.add_argument('--pos_hid_dim', default=200, type=int)
    parser.add_argument('--word_hid_dim', default=200, type=int)
    parser.add_argument('--pos_num_layers', default=1, type=int)
    parser.add_argument('--word_num_layers', default=1, type=int)
    parser.add_argument('--pos_gate', action='store_true')
    parser.add_argument('--word_gate', action='store_true')
    # train
    parser.add_argument('--pos_weight', default=0.5, type=float)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hooks_per_epoch', default=1, type=int)
    parser.add_argument('--checkpoints', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.load_dataset:
        dataset = BlockDataset.from_disk(args.dataset_path)
        dataset.set_batch_size(args.batch_size), dataset.set_gpu(args.gpu)
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
    train, valid = dataset.splits(test=None)

    pos_dict, word_dict = train.d

    m = ChainPOSAwareLM(
        (len(pos_dict.vocab), len(word_dict.vocab)),  # vocabs
        (args.pos_emb_dim, args.word_emb_dim),
        (args.pos_hid_dim, args.word_hid_dim),
        num_layers=(args.pos_num_layers, args.word_num_layers),
        pos_gate=args.pos_gate, word_gate=args.word_gate,
        dropout=args.dropout)

    m.apply(u.make_initializer())

    print(m)

    if args.gpu:
        m.cuda(), train.set_gpu(args.gpu), valid.set_gpu(args.gpu)

    crit = make_pos_word_criterion(gpu=args.gpu)
    optim = Optimizer(m.parameters(), args.optim, lr=args.lr)
    trainer = POSAwareLMTrainer(
        m, {'train': train, 'valid': valid}, crit, optim,
        pos_weight=args.pos_weight)
    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_generate_hook(pos_dict, word_dict),
                     hooks_per_epoch=args.hooks_per_epoch)
    trainer.train(args.epochs, args.checkpoints)
