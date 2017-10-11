
import os
import time
import random

import seqmod.utils as u


def normalize(scores, hyps, lang_d):
    total_score, doc = 0, []
    for score, hyp in zip(scores, hyps):
        total_score += score
        hyp = ''.join([lang_d.vocab[c] for c in hyp])
        hyp = hyp.split(u.EOS)[1: -1]
        doc.extend(hyp)
    total_score /= len(hyps)

    return total_score, doc


def generate_clm(model, lang_d, conds_d, conds, **kwargs):
    if len(conds) != len(conds_d):
        raise ValueError(f"Model requires {len(conds_d)} input conditions")

    if kwargs.get('gpu'):
        model.cuda()

    # random sample of conds if not passed
    for idx, (d, c) in enumerate(zip(conds_d, conds)):
        if c is None:
            conds[idx] = d.index(random.choice(d.vocab))
        else:
            conds[idx] = d.index(c)

    # generate
    scores, hyps = model.generate(
        lang_d, conds=conds, ignore_eos=True, **kwargs)

    # normalize
    return normalize(scores, hyps, lang_d)


def generate_lm(model, lang_d, **kwargs):
    if kwargs.get('gpu'):
        model.cuda()

    # generate
    scores, hyps = model.generate(lang_d, ignore_eos=True, **kwargs)

    # normalize
    return normalize(scores, hyps, lang_d)


def generate_from_dict(indir, outputdir, **kwargs):
    # get model path
    fname = None
    for f in os.listdir(indir):
        if f.startswith('clm'):
            fname = '.'.join(f.split(".")[:-1])
            if fname.endswith('dict'):
                fname = '.'.join(fname.split(".")[:-1])
            fname = os.path.join(indir, fname)
    if fname is None:
        raise ValueError("Couldn't find model path")

    m = u.load_model(fname + '.pt')
    (lang_d, *conds_d), _ = u.load_model(fname + '.dict.pt')

    # assume author dict is the first cond dict
    authors = conds_d[0].vocab
    for author in authors:

        print(f"Generating author {author}")

        # default to longest sentence
        score, doc = generate_clm(m, lang_d, conds_d, [author, '300'], **kwargs)

        with open(os.path.join(outputdir, f'{author}.txt'), 'w+') as f:
            for line in doc:
                f.write(line)


def generate_from_dir(indir, outputdir, **kwargs):
    # compute model paths
    paths = set()

    for f in os.listdir(indir):
        fname = os.path.join(indir, f)
        if not fname.endswith('dict.pt'):
            continue
        paths.add('.'.join(fname.split('.')[:-2]))

    for fname in paths:

        author = os.path.basename(fname).split('_')[0]
        print(f"Generating author {author}")

        mpath, dpath = fname + '.pt', fname + '.dict.pt'
        if not (os.path.isfile(mpath) and os.path.isfile(dpath)):
            raise ValueError(f"Missing dict or model for file [{fname}]")

        m, lang_d = u.load_model(mpath), u.load_model(dpath)
        score, doc = generate_lm(m, lang_d, **kwargs)

        with open(os.path.join(outputdir, f'{author}.txt'), 'w+') as f:
            for line in doc:
                f.write(line + '\n')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir')
    parser.add_argument('outputdir')
    parser.add_argument('--model', help='One of (clm,lm)', required=True)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--max_seq_len', default=1000, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    indir = os.path.join(args.rootdir, 'models', args.model)
    if not os.path.isdir(indir):
        raise ValueError(f"Input directory {indir} doesn't exist")
    outputdir = os.path.join(args.outputdir, args.model)
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    func = generate_from_dir if args.model == 'lm' else generate_from_dict
    func(indir, outputdir, batch_size=args.batch_size, gpu=args.gpu,
         max_seq_len=args.max_seq_len)
