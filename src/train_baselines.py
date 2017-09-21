
import os
import json
import pprint

from sklearn.externals import joblib
from sklearn import preprocessing

from utils import test_report, readlines, get_author
from baselines import svm_search, grid_report


def load_dataset(path):
    X, y = [], []
    for f in os.listdir(path):
        doc = []
        for _, line in readlines(os.path.join(path, f)):
            doc.append(line.strip())
        X.append(doc), y.append(get_author(f))
    return X, y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--outputpath')
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.path, 'alpha')) or \
       not os.path.isdir(os.path.join(args.path, 'omega')) or \
       not os.path.isdir(os.path.join(args.path, 'alpha_bar')):
        raise ValueError("Missing 'alpha', 'alpha_bar' or 'omega' paths")

    # train pipeline on alpha & omega & alpha_bar
    dirs = ('alpha', 'omega', 'alpha_bar')
    for d in dirs:
        X, y = load_dataset(os.path.join(args.path, d))
        le = preprocessing.LabelEncoder().fit(y)
        grid = svm_search(X, le.transform(y), n_iter=args.n_iter, cv=args.cv)

        print("Saving grid for {}".format(d))
        joblib.dump(grid, os.path.join(args.outputpath, f'{d}.grid.pk'))

        print("CV output for {}".format(d))
        grid_report(grid.cv_results_)

        for dd in dirs:
            if d == dd:
                continue
            print(f"Testing grid {d} on {dd}")
            X_test, y_test = load_dataset(os.path.join(args.path, dd))
            y_pred = grid.predict(X_test)

        report = test_report(le.transform(y_test), y_pred, le)
        pprint(report)
        reportfile = os.path.join(args.outputpath, f'{d}.{dd}.report.json')
        with open(reportfile, 'w') as f:
            json.dump(report, reportfile)
