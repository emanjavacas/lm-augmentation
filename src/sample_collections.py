
import os
import random
from collections import defaultdict

from preprocess import Reader, make_tokenizer, writelines


def get_author(fname):
    return os.path.basename(fname).split('_')[0]


def sample_files(path, nb_docs, nb_words):
    wc, omega, alpha = defaultdict(dict), [], []
    total, files = nb_docs * nb_words, list(os.listdir(path))
    for f in files:
        f = os.path.join(path, f)
        with open(f, 'r') as inf:
            wc[get_author(f)][f] = len(inf.read().split())
    for author, works in wc.items():
        if len(works) < 2:
            print("Not enough works by author {}".format(author))
            continue
        works = sorted(works.items(), key=lambda x: x[1])
        (a_fs, a_cs), (b_fs, b_cs) = zip(*works[::2]), zip(*works[1::2])
        if sum(a_cs) < total or sum(b_cs) < total:
            print("Omitting author {}, not enough words".format(author))
            continue
        if random.random() > 0.5:
            omega.append(a_fs), alpha.append(b_fs)
        else:
            omega.append(b_fs), alpha.append(a_fs)
    return alpha, omega


def sample_split(files, nb_docs, nb_words, outputpath='output', lang='eng'):
    for fs in files:
        author, fs = get_author(fs[0]), list(fs)
        sents = Reader(fs, make_tokenizer(lang)).process(shuffle_pars=True)
        doc_length, doc, docs = 0, [], 0
        while docs < nb_docs:
            for labels, sent in sents:
                doc_length += len(sent.split())
                doc.append((labels, sent))
                if doc_length >= nb_words:
                    filename = '{}_{}'.format(author, docs + 1)
                    writelines(os.path.join(outputpath, filename), doc)
                    doc_length, doc, docs = 0, [], docs + 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--outputpath')
    parser.add_argument('--nb_docs', type=int, default=10)
    parser.add_argument('--nb_words', type=int, default=10000)
    parser.add_argument('--lang', default='eng')
    args = parser.parse_args()

    alpha, omega = sample_files(args.path, args.nb_docs, args.nb_words)
    sample_split(alpha, args.nb_docs, args.nb_words,
                 outputpath=args.outputpath, lang=args.lang)
