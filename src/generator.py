
import time
import random

import seqmod.utils as u


class CLMGenerator:
    def __init__(self, model, lang_d, conds_d):
        self.model = model
        self.lang_d = lang_d
        self.conds_d = conds_d

    def generate_doc(self, conds, max_seq_len=1000, batch_size=10, **kwargs):
        if len(conds) != len(self.conds_d):
            raise ValueError
        conds = [d.index(random.choice(d.vocab) if c is None else c)
                 for (d, c) in zip(self.conds_d, conds)]
        scores, hyps = self.model.generate(
            self.lang_d, conds=conds, max_seq_len=max_seq_len,
            batch_size=batch_size, ignore_eos=True, **kwargs)
        total_score, doc = 0, []
        for score, hyp in zip(scores, hyps):
            total_score += score
            hyp = ''.join([self.lang_d.vocab[c] for c in hyp])
            hyp = hyp.split(u.EOS)[1: -1]
            doc.extend(hyp)
        total_score /= len(hyps)
        return total_score, doc


if __name__ == '__main__':
    m = u.load_model("../models/cLSTM-l1-h2048-e48-b150-2.633.pt")
    model, (lang_d, *conds_d) = m['model'], m['d']
    model.cuda()
    generator = CLMGenerator(model, lang_d, conds_d)
    start = time.time()
    out = generator.generate_doc(['Cbronte', None], gpu=True, batch_size=100)
    print(time.time() - start)
