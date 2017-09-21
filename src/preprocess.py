
import os
import warnings
import random; random.seed(1001)
from collections import Counter

from utils import compute_length, writelines

try:
    import ucto
except ImportError:
    warnings.warn("Couldn't import ucto; using dummy tokenizer")
    ucto = None


def make_tokenizer(lang):
    if ucto is not None:
        return ucto.Tokenizer("tokconfig-{lang}".format(lang=lang),
                              paragraphdetection=True)


def split_sentences(par, tokenizer):
    prev_ending = [], False
    if tokenizer is not None:
        tokenizer.process(par)
        sentence = ''
        for token in tokenizer:
            if prev_ending and token.isbeginofsentence():
                sentence = sentence.strip()
                if sentence:
                    yield sentence
                    sentence = ''
            prev_ending = token.isendofsentence()
            sentence += str(token)
            if not token.nospace():
                sentence += ' '
        if sentence:
            sentence = sentence.strip()
            yield sentence
    else:
        for line in par.split('\n'):
            yield line.split()


def split_pars(filename):
    buf, prev_par = '', False
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if prev_par:
                    continue
                yield buf
                buf, prev_par = '', True
            else:
                buf += line + '\n'
                prev_par = False


class Reader(object):
    def __init__(self, fileiter, tokenizer):
        if isinstance(fileiter, str):
            self.fileiter = [os.path.join(fileiter, f)
                             for f in os.listdir(fileiter)]
        else:
            self.fileiter = fileiter
        self.tokenizer = tokenizer

    def on_filename(self, filename):
        "Callback to extract sentence labels from filename"
        return [filename.split('_')[0]]

    def on_par(self, par):
        "Callback to extract sentence labels at the paragraph level"
        return []

    def on_sent(self, sent):
        "Callback to extract sentence labels from the sentence itself"
        return [compute_length(sent)]

    def sent_length(self, sent):
        "Compute sentence length. Default implementation uses words"
        return len(sent)

    def process(self, shuffle_files=True, shuffle_pars=False,
                balance_ref=None, balance_count=None):
        """
        Generator over labeled sentences.

        Parameters:
        -----------
        balance_ref: None or int, integer pointing to the position in the
            sentence label array of the label to be used for balancing
        balance_count: None or int, maximum count per balancing label
        """
        if shuffle_files:
            random.shuffle(self.fileiter)
        pars = []
        for f in self.fileiter:
            file_labels = self.on_filename(os.path.basename(f))
            for par in split_pars(f):
                par_labels = self.on_par(par)
                pars.append((file_labels + par_labels, par))
        if shuffle_pars:
            random.shuffle(pars)
        if balance_ref is not None and balance_count is not None:
            balance_counter = Counter(int)
        for labels, par in pars:
            for sent in split_sentences(par, self.tokenizer):
                sent_labels = labels + self.on_sent(sent)
                if balance_ref is not None and balance_count is not None:
                    ref = sent_labels[balance_ref]
                    balance_counter[ref] += self.sent_length(sent)
                    if balance_counter[ref] > balance_count:
                        continue
                yield sent_labels, sent


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute train, test splits for an input corpus")
    parser.add_argument('corpus')
    parser.add_argument('outputdir')
    parser.add_argument('--train_size', default=0.75, type=float)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--lang', default='eng')
    args = parser.parse_args()

    reader = Reader(args.corpus, make_tokenizer(args.lang))
    rows = list(reader.process(shuffle_files=True, shuffle_pars=True))
    split = int(len(rows) * args.train_size)
    train_rows, test_rows = rows[:split], rows[-split:]

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    writelines(os.path.join(args.outputdir, 'train'), train_rows)
    writelines(os.path.join(args.outputdir, 'test'), test_rows)
