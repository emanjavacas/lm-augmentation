
import os
import csv

try:
    import ucto
except ImportError:
    print("Couldn't import ucto; using dummy tokenizer")
    ucto = None

from sklearn.model_selection import train_test_split


def make_tokenizer(lang):
    if ucto is not None:
        return ucto.Tokenizer("tokconfig-{lang}".format(lang=lang),
                              paragraphdetection=True)


def split_sentences(pars, tokenizer, lang):
    sentences, prev_ending = [], False
    for par in pars:
        if tokenizer is not None:
            tokenizer.process(par)
            sentence = ''
            for token in tokenizer:
                if prev_ending and token.isbeginofsentence():
                    sentence = sentence.strip()
                    if sentence:
                        sentences.append(sentence)
                        sentence = ''
                prev_ending = token.isendofsentence()
                sentence += str(token)
                if not token.nospace():
                    sentence += ' '
            if sentence:
                sentence = sentence.strip()
                sentences.append(sentence)
            yield sentences
            sentences = []
        else:
            for line in par.split('\n'):
                sentences.append(line.split())
            yield sentences
            sentences = []


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


def make_fileiter(path, class_extractor=lambda f: f.split('_')[0]):
    for f in os.listdir(path):
        yield os.path.join(path, f), class_extractor(f)


def paragraphs(fileiter, lang='eng'):
    par, tokenizer = [], make_tokenizer(lang)
    for f, label in fileiter:
        for par in split_sentences(split_pars(f), tokenizer, lang):
            yield par, label


def writepars(outputfile, pars, labels):
    with open(outputfile + '.csv', 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for par, label in zip(pars, labels):
            row = [label, r'\n'.join(par)]
            csvwriter.writerow(row)


def readpars(inputfile):
    with open(inputfile, 'r', newline='\n') as f:
        for line in f:
            label, par = line.split('\t')
            yield label, par.split('\\n')


if __name__ == '__main__':
    import argparse
    import random
    parser = argparse.ArgumentParser(
        description="Compute train, test splits for an input corpus")
    parser.add_argument('corpus')
    parser.add_argument('outputdir')
    parser.add_argument('--train_size', default=0.75, type=float)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--lang', default='eng')
    args = parser.parse_args()

    random.seed(args.seed)

    pars = list(paragraphs(make_fileiter(args.corpus), lang=args.lang))
    random.shuffle(pars)
    pars, labels = zip(*pars)
    train_pars, test_pars, train_labels, test_labels = train_test_split(
        pars, labels,
        train_size=args.train_size, stratify=labels,
        random_state=args.seed)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    writepars(os.path.join(args.outputdir, 'train'), train_pars, train_labels)
    writepars(os.path.join(args.outputdir, 'test'), test_pars, test_labels)
