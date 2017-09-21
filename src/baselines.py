
import pprint
import multiprocessing

import numpy as np

from sklearn import svm, preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

from process import readpars
from utils import test_report


def pars_to_X(inputfile):
    X, y = [], []
    for label, lines in readpars(inputfile):
        y.append(label), X.append('\r\n'.join(lines))
    return X, y


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


def todense(X): return X.todense()


def svm_search(X_train, y_train, n_iter=100, seed=1001, cv=5):
    pipe = Pipeline(
        [('vectorizer', TfidfVectorizer()),
         ('to_dense', preprocessing.FunctionTransformer(
             todense, accept_sparse=True)),
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
        pipe, param_space,
        error_score=0.0,        # return score on error
        scoring='f1_micro',     # SVM defaults to accuracy otherwise
        n_jobs=max(multiprocessing.cpu_count() - 2, 1),  # don't hog
        verbose=1,
        cv=cv,
        n_iter=n_iter,
        random_state=seed)

    grid.fit(X_train, y_train)

    return grid

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('trainfile')
    parser.add_argument('testfile')
    parser.add_argument('--outputfile')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    X_train, y_train = pars_to_X(args.trainfile)
    le = preprocessing.LabelEncoder().fit(y_train)
    grid = svm_search(X_train, le.transform(y_train), n_iter=args.n_iter, cv=args.cv)

    grid_report(grid.cv_results_)

    X_test, y_test = pars_to_X(args.testfile)
    y_pred = grid.predict(X_test)

    pprint(test_report(le.transform(y_test), y_pred, le))

    if args.outputfile:
        joblib.dump(grid, args.outputfile)
