import numpy as np

from classifier_search import best_classifier
from composite_regressor import CompositeRegressor
from featurize_training_set import load_featurized_training_set

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


def compare_regressors(filename):
    clf_simple = SGDRegressor()
    clf_composite = CompositeRegressor(best_classifier(), SGDRegressor())

    features = load_featurized_training_set(filename)
    X = features.drop(['DATE', 'n_calls'], axis=1).as_matrix().astype(float)
    X = StandardScaler().fit_transform(X)
    y = features.n_calls.as_matrix()

    mse_compsite = []
    mse_simple = []

    classifiers = {"simple": clf_simple, "composite": clf_composite}
    mse = {"simple": [], "composite": []}

    kf = KFold(len(X), n_folds=5, shuffle=True)
    for train, test in kf:
        for type in classifiers:
            clf = classifiers[type]
            clf.fit(X[train], y[train])
            predicted = clf.predict(X[test])
            mse[type].append(mean_squared_error(y[test], predicted))

    print("Average for simple classifier: " + str(np.mean(mse['simple'])))
    print("Average for composite classifier: " + str(np.mean(mse['composite'])))


if __name__ == "__main__":
    compare_regressors("files/train_featurized.pkl")
