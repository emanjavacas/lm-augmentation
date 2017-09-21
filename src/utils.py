
import os
import csv
import json
import math

from sklearn.metrics import precision_recall_fscore_support


def get_author(fname):
    return os.path.basename(fname).split('_')[0]


def writelines(outputfile, rows):
    with open(outputfile + '.csv', 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for ls, sent in rows:
            csvwriter.writerow([*ls, sent])


def readlines(inputfile):
    with open(inputfile, 'r', newline='\n') as f:
        for line in f:
            *labels, sent = line.split('\t')
            yield labels, sent


def compute_length(l, length_bins=(50, 100, 150, 300)):
    length = len(l)
    output = None
    for length_bin in length_bins[::-1]:
        if length > length_bin:
            output = length_bin
            break
    else:
        output = -1
    return output


def test_report(y_true, y_pred, le=None):
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    labels = list(set(y_true))
    report = []
    for i in range(len(labels)):
        if le is not None:
            label = le.inverse_transform(i)
        else:
            label = labels[i]
        report.append(
            {'label': label,
             'result': {'precision': p[i],
                        'recall': r[i],
                        'f1': f1[i],
                        'support': int(s[i])}})
    return report


def get_trues_preds(outputfile):
    trues, preds, lengths = [], [], []
    with open(outputfile, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            argmax, maxpred = None, -math.inf
            for author, pred in line['preds'].items():
                if float(pred) > maxpred:
                    argmax = author
                    maxpred = float(pred)
            preds.append(argmax)
            trues.append(line['true'])
            lengths.append(line['length'])
    return trues, preds


def get_trues_ranked_preds(outputfile):
    trues, preds, lengths = [], [], []
    with open(outputfile, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            ranked = sorted(
                line['preds'].items(), key=lambda item: float(item[1]),
                reverse=True)
            preds.append([label for label, _ in ranked])
            trues.append(line['true'])
            lengths.append(line['length'])
    return trues, preds
