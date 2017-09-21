
import multiprocessing

import numpy as np

from sklearn import svm, preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


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


def svm_search(X, y, n_iter=100, seed=1001, cv=5):
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

    grid.fit(X, y)

    return grid
