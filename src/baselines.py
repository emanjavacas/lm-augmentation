
import os
import json
from pprint import pprint
import multiprocessing

from collections import defaultdict

import numpy as np

from sklearn.externals import joblib
from sklearn import svm
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import test_report, readlines, get_author


def load_dataset(path, authors=None):
    X, y = [], []
    for f in os.listdir(path):
        author = get_author(f)
        if authors is not None and author not in authors:
            print(f"Skipping author {author}")
            continue
        doc = ""
        if f.endswith(".csv"):
            for _, line in readlines(os.path.join(path, f)):
                doc += line
        else:
            with open(os.path.join(path, f), 'r') as f:
                for line in f:
                    doc += line.strip()
        X.append(doc), y.append(author)
    return X, y


def sample_docs(f, nb_words):
    with open(f, 'r+') as f:
        doc = ""
        for line in f:
            if len(doc.split()) >= nb_words:
                yield doc
                doc = ""
            doc += line


def make_sampler(root, nb_words):
    out = {}
    for f in os.listdir(root):
        author = f.split('.')[0]
        out[author] = sample_docs(os.path.join(root, f), nb_words)
    return out


def add_documents(X, y, sampler, nb_docs):
    for author, docs in sampler.items():
        new_docs = 0
        for doc in docs:
            new_docs += 1
            X.append(doc), y.append(author)
            if new_docs >= nb_docs:
                break


def grid_report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def to_dense(X): return X.todense()


def svm_search(X, y, n_iter=100, seed=1001, cv=5):
    pipe = Pipeline(
        [('vectorizer', TfidfVectorizer()),
         ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
         ('classifier', svm.SVC())])

    param_space = {
        'vectorizer': [TfidfVectorizer(analyzer='char', ngram_range=(2, 4))],
        'vectorizer__use_idf': [True, False],
        'vectorizer__max_features': [5000, 10000, 15000, 30000],
        'vectorizer__norm': ['l1', 'l2'],
        'classifier__C': [1, 10, 100, 1000],
        'classifier__kernel': ['linear', 'rbf']
    }

    grid = RandomizedSearchCV(
        pipe,
        param_space,
        error_score=0.0,        # return score on error
        scoring='f1_micro',     # SVM defaults to accuracy otherwise
        n_jobs=max(multiprocessing.cpu_count() - 2, 1),  # don't hog
        verbose=1,
        cv=cv,
        n_iter=n_iter,
        random_state=seed
    )

    grid.fit(X, y)

    return grid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--outputpath', required=True)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--model', help='One of lm, clm')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.path, 'alpha')) or \
       not os.path.isdir(os.path.join(args.path, 'omega')):
        raise ValueError("Missing 'alpha', 'omega' paths")

    # train pipeline on alpha & omega & alpha_bar
    if not args.augmentation:

        dirs = ('alpha', 'omega')
        for d in dirs:

            # fit grid
            X, y = load_dataset(os.path.join(args.path, d))
            le = LabelEncoder().fit(y)
            grid = svm_search(
                X, le.transform(y), n_iter=args.n_iter, cv=args.cv)
            # save grid
            grid_output_file = os.path.join(args.outputpath, f'{d}.grid.pk')
            print(f"Saving grid for {d} on {grid_output_file}")
            joblib.dump(grid, grid_output_file)
            # save CV report
            print("CV output for {}".format(d))
            grid_report(grid.cv_results_)

            # test on other datasets
            for dd in dirs:
                if d == dd:
                    continue

                print(f"Testing grid {d} on {dd}")
                X_test, y_test = load_dataset(
                    os.path.join(args.path, dd), authors=set(y))
                y_pred = grid.predict(X_test)

                print(f"Test Report for {d}-{dd}")
                report = test_report(le.transform(y_test), y_pred, le)
                pprint(report)
                reportfile = os.path.join(
                    args.outputpath, f'{d}.{dd}.report.json')
                with open(reportfile, 'w') as f:
                    json.dump(report, reportfile)

    # Do augmentation (requires alpha_bar)
    else:

        alpha_bar_path = os.path.join(args.path, 'alpha_bar', args.model)
        if not os.path.isdir(alpha_bar_path):
            raise ValueError("Missing 'alpha_bar' directory")

        # load datasets
        omega_X, omega_y = load_dataset(os.path.join(args.path, 'omega'))
        alpha_X, alpha_y = load_dataset(os.path.join(args.path, 'alpha'))

        le = LabelEncoder().fit(omega_y + alpha_y)

        # estimate nb_words, nb_docs
        nb_docs, nb_words = defaultdict(int), defaultdict(int)
        for doc, y in zip(alpha_X, alpha_y):
            nb_words[y] += len(doc.split())
            nb_docs[y] += 1
        nb_words = sum(nb_words.values()) // sum(nb_docs.values())
        nb_docs = sum(nb_docs.values()) // len(nb_docs)

        # sample_docs
        alpha_bar_sampler = make_sampler(alpha_bar_path, nb_words)
        alpha_bar_X, alpha_bar_y = [], []

        for breakpoint in range(nb_docs // 2, nb_docs * 5, 5):
            print(f"Breakpoint: {breakpoint}")
            new_docs = breakpoint - len(alpha_bar_X)
            add_documents(
                alpha_bar_X, alpha_bar_y, alpha_bar_sampler, new_docs)

            grid = svm_search(
                alpha_bar_X, le.transform(alpha_bar_y),
                n_iter=args.n_iter, cv=args.cv)

            grid_output_file = os.path.join(
                args.outputpath,
                f'augmented_{breakpoint}.{args.model}.grid.pk')
            print(f"Saving grid for bp {breakpoint} on {grid_output_file}")
            joblib.dump(grid, grid_output_file)

            print("CV output for {breakpoint}")
            grid_report(grid.cv_results_)

            y_pred = grid.predict(omega_X)
            report = test_report(le.transform(omega_y), y_pred, le)
            pprint(report)
            reportfile = os.path.join(
                args.outputpath, f'augmented_{breakpoint}.{model}.report.json')
            with open(reportfile, 'w') as f:
                json.dump(report, reportfile)
