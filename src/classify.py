
import math
import time
import json

import torch
from torch.autograd import Variable

import seqmod.utils as u

from utils import compute_length, readlines


class CLMClassifier(object):
    def __init__(self, path, gpu=False):
        m = u.load_model(path)
        self.gpu = gpu
        self.model = m['model']
        self.model.eval()
        if self.gpu:
            self.model.cuda()
        (self.lang_d, self.author_d, self.lengths_d) = m['d']

    def classify(self, doc):
        # (1) prepare data
        n_authors = len(self.author_d)
        chars, lengths = [], []
        for line in doc:
            length = self.lengths_d.index(compute_length(line))
            for char in next(self.lang_d.transform([line])):
                chars.append(char), lengths.append(length)
        inp = torch.LongTensor(chars).unsqueeze(1).repeat(1, n_authors)
        lengths = torch.LongTensor(lengths).unsqueeze(1).repeat(1, n_authors)
        authors = []
        for author in self.author_d.vocab:
            author = [self.author_d.index(author)] * len(inp)
            authors.append(torch.LongTensor(author))
        authors = torch.stack(authors, 1)
        if self.gpu:
            inp, authors, lengths = inp.cuda(), authors.cuda(), lengths.cuda()

        # (2) run prediction
        logits = self.model(
            Variable(inp, volatile=True),
            conds=(Variable(authors, volatile=True),
                   Variable(lengths, volatile=True)))[0]
        # move to cpu for the prediction and -> (authors x seq_len x vocab)
        seq_len = len(chars)
        logits = logits.cpu().view(seq_len, n_authors, -1).t().data
        preds = {}
        for n_author, author_logits in enumerate(logits):
            author = self.author_d.vocab[n_author]
            true_logits = u.select_cols(author_logits[:-1], chars[1:])
            preds[author] = math.exp((true_logits / len(true_logits)).sum())
        return preds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('modelfile')
    parser.add_argument('--outputfile', default='output.json')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    lines = readlines(args.inputfile)
    clf = CLMClassifier(args.modelfile, gpu=args.gpu)
    labels, lines = zip(*lines)

    print("Predicting {} lines".format(len(lines)))
    start = time.time()
    for idx, (label, line) in enumerate(zip(labels, lines[300:])):
        linelength = len(''.join(line))
        if args.gpu and linelength > 4000:  # don't classify too long pars OOE
            print("Omitting too long document {}".format(idx))
            continue
        print("linelength {}".format(linelength))
        preds = clf.classify(line)
        with open(args.outputfile, 'a') as f:
            preds = {'preds': preds, 'true': label, 'length': linelength}
            jsonstr = json.dumps(preds)
            f.write(jsonstr + '\n')
    print("Took {}".format(time.time() - start))
